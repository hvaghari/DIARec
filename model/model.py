"""DIARec reference implementation (TensorFlow/Keras).

Architecture follows the paper:
DIARec: Dynamic Intention-Aware Recommendation with Attention-Based
Context-Aware Item Attributes Modeling (JAISCR, 2024).

This implementation is designed as a clear, inspectable reference for GitHub,
architecture reporting, parameter counting, and complexity profiling.

Input convention
----------------
Padding item id is 0. Real item ids should start from 1.

inputs = {
    "user_id":                 int32 [B],
    "last_item_id":            int32 [B],
    "long_attr_item_ids":      int32 [B, L, A, N],
    "long_attr_weight_class":  int32 [B, L, A],
    "short_attr_item_ids":     int32 [B, A, N],
    "short_attr_weight_class": int32 [B, A],
    "short_sequence_ids":      int32 [B, T],
}

where:
    B = batch size
    L = number of long-term sessions
    A = number of attribute-relation types
    N = maximum number of items associated with one attribute in a session
    T = maximum current-session sequence length

The five attribute-weight classes are encoded as integer ids 0..4.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


@dataclass
class DIARecConfig:
    num_users: int
    num_items: int
    num_attributes: int
    num_weight_classes: int = 5
    embedding_dim: int = 16
    num_heads: int = 3
    attention_dim: int = 30
    gru_units: int = 16
    fusion_dim: int = 30
    dropout: float = 0.5
    l2_attention: float = 0.1
    l2_fusion: float = 0.01
    l2_output: float = 0.1

    def __post_init__(self):
        if self.num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if self.attention_dim % self.num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        if self.num_weight_classes != 5:
            # The paper defines five percentage-based attribute-weight classes.
            raise ValueError("DIARec paper defines exactly 5 attribute-weight classes")


class ContextualItemAttention(layers.Layer):
    """Equations (1)-(3): user-guided attention over items in V_a^u."""

    def __init__(self, hidden_dim: int, dropout: float, l2_reg: float, **kwargs):
        super().__init__(**kwargs)
        reg = regularizers.l2(l2_reg)
        self.item_proj = layers.Dense(hidden_dim, use_bias=False, kernel_regularizer=reg)
        self.user_proj = layers.Dense(hidden_dim, use_bias=True, kernel_regularizer=reg)
        self.score = layers.Dense(1, activation="relu", kernel_regularizer=reg)
        self.dropout = layers.Dropout(dropout)

    def call(self, user_emb, item_embs, item_mask=None, training=None):
        # user_emb: [..., D], item_embs: [..., N, D]
        q = self.user_proj(user_emb)[..., tf.newaxis, :]
        k = self.item_proj(item_embs)
        logits = tf.squeeze(self.score(k + q), axis=-1)  # [..., N]

        if item_mask is not None:
            very_negative = tf.cast(-1e9, logits.dtype)
            logits = tf.where(item_mask, logits, very_negative)

        beta = tf.nn.softmax(logits, axis=-1)
        beta = self.dropout(beta, training=training)
        context = tf.reduce_sum(item_embs * beta[..., tf.newaxis], axis=-2)
        return context, beta


class CollaborativeAttributeAttention(layers.Layer):
    """Collaborative multi-head contextual attribute attention (Eq. 4-8).

    Each head first computes an attribute score from:
      key = attribute_embedding * attribute_weight_embedding
      query = concat(user_embedding, last_item_embedding)

    Before softmax, a trainable h x h collaborative projection mixes scores
    across the HEAD dimension, matching the role of delta_ij in Eq. (5).
    Values are the item-level attribute representations V_a^u from Eq. (3).
    """

    def __init__(
        self,
        num_heads: int,
        attention_dim: int,
        output_dim: int,
        dropout: float,
        l2_reg: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.output_dim = output_dim
        reg = regularizers.l2(l2_reg)

        self.key_proj = layers.Dense(attention_dim, use_bias=False, kernel_regularizer=reg)
        self.query_proj = layers.Dense(attention_dim, use_bias=True, kernel_regularizer=reg)
        self.head_score = layers.Dense(num_heads, activation="relu", kernel_regularizer=reg)
        self.output_proj = layers.Dense(output_dim, use_bias=False, kernel_regularizer=reg)
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        # delta_ij: information-sharing weights among attention heads.
        self.delta = self.add_weight(
            name="collaborative_projection",
            shape=(self.num_heads, self.num_heads),
            initializer=keras.initializers.Identity(gain=1.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(
        self,
        user_emb,
        last_item_emb,
        attribute_embs,
        weight_embs,
        attribute_values,
        attribute_mask=None,
        training=None,
    ):
        # user_emb,last_item_emb: [..., D]
        # attribute_embs,weight_embs,attribute_values: [..., A, D]
        key = attribute_embs * weight_embs
        query = tf.concat([user_emb, last_item_emb], axis=-1)

        k = self.key_proj(key)
        q = self.query_proj(query)[..., tf.newaxis, :]
        hidden = tf.nn.relu(k + q)

        # Independent pre-projection head scores: [..., A, H]
        raw_scores = self.head_score(hidden)

        # Eq. (5): mix across the HEAD axis, not the attribute/sequence axis.
        mixed_scores = tf.einsum("...ah,hk->...ak", raw_scores, self.delta)
        logits = tf.linalg.matrix_transpose(mixed_scores)  # [..., H, A]

        if attribute_mask is not None:
            mask = attribute_mask[..., tf.newaxis, :]  # [..., 1, A]
            very_negative = tf.cast(-1e9, logits.dtype)
            logits = tf.where(mask, logits, very_negative)

        alpha = tf.nn.softmax(logits, axis=-1)
        alpha = self.dropout(alpha, training=training)

        # Eq. (7): each head aggregates V_a^u using its attribute weights.
        head_context = tf.einsum("...ha,...ad->...hd", alpha, attribute_values)
        concat_heads = tf.reshape(
            head_context,
            tf.concat([tf.shape(head_context)[:-2], [self.num_heads * tf.shape(head_context)[-1]]], axis=0),
        )

        # Eq. (8): W^O Concat(I^1,...,I^h)
        intention = self.output_proj(concat_heads)
        return intention, alpha


class IntentionAwareUnit(layers.Layer):
    """Hierarchical IAU: contextual item attention -> attribute attention."""

    def __init__(self, config: DIARecConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.item_attention = ContextualItemAttention(
            hidden_dim=config.embedding_dim,
            dropout=config.dropout,
            l2_reg=config.l2_attention,
            name="contextual_item_attention",
        )
        self.attribute_attention = CollaborativeAttributeAttention(
            num_heads=config.num_heads,
            attention_dim=config.attention_dim,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
            l2_reg=config.l2_attention,
            name="collaborative_attribute_attention",
        )

    def call(
        self,
        user_emb,
        last_item_emb,
        attr_item_embs,
        attribute_embs,
        weight_embs,
        item_mask=None,
        attribute_mask=None,
        training=None,
    ):
        # attr_item_embs: [..., A, N, D]
        # Broadcast user over A.
        a = tf.shape(attr_item_embs)[-3]
        user_for_attr = tf.broadcast_to(
            user_emb[..., tf.newaxis, :],
            tf.concat([tf.shape(user_emb)[:-1], [a, tf.shape(user_emb)[-1]]], axis=0),
        )

        values, beta = self.item_attention(
            user_for_attr, attr_item_embs, item_mask=item_mask, training=training
        )  # [..., A, D]

        intention, alpha = self.attribute_attention(
            user_emb=user_emb,
            last_item_emb=last_item_emb,
            attribute_embs=attribute_embs,
            weight_embs=weight_embs,
            attribute_values=values,
            attribute_mask=attribute_mask,
            training=training,
        )
        return intention, {"item_attention": beta, "attribute_attention": alpha}


class TemporalAttention(layers.Layer):
    """Equations (12)-(14): current intention queries encoded long-term sessions."""

    def __init__(self, hidden_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.history_proj = layers.Dense(hidden_dim, use_bias=False)
        self.current_proj = layers.Dense(hidden_dim, use_bias=True)
        self.score = layers.Dense(1, activation="relu")
        self.dropout = layers.Dropout(dropout)

    def call(self, history, current_intention, session_mask=None, training=None):
        # history: [B,L,H], current_intention: [B,D]
        logits = tf.squeeze(
            self.score(
                self.history_proj(history)
                + self.current_proj(current_intention)[:, tf.newaxis, :]
            ),
            axis=-1,
        )
        if session_mask is not None:
            logits = tf.where(session_mask, logits, tf.cast(-1e9, logits.dtype))
        eta = tf.nn.softmax(logits, axis=-1)
        eta = self.dropout(eta, training=training)
        long_short = tf.reduce_sum(history * eta[..., tf.newaxis], axis=1)
        return long_short, eta


class DIARec(keras.Model):
    """DIARec reference architecture."""

    def __init__(self, config: DIARecConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        c = config

        # +1 reserves item/user id 0 for padding/unknown where applicable.
        self.user_embedding = layers.Embedding(c.num_users + 1, c.embedding_dim, name="user_embedding")
        self.item_embedding = layers.Embedding(c.num_items + 1, c.embedding_dim, mask_zero=True, name="item_embedding")
        self.attribute_embedding = layers.Embedding(c.num_attributes, c.embedding_dim, name="attribute_embedding")
        self.weight_embedding = layers.Embedding(c.num_weight_classes, c.embedding_dim, name="weight_embedding")

        self.iau = IntentionAwareUnit(c, name="intention_aware_unit")

        self.long_encoder = layers.Bidirectional(
            layers.GRU(c.gru_units, return_sequences=True),
            merge_mode="concat",
            name="long_term_bigru",
        )
        self.temporal_attention = TemporalAttention(
            hidden_dim=c.embedding_dim,
            dropout=c.dropout,
            name="time_aware_attention",
        )
        self.short_encoder = layers.GRU(c.gru_units, name="short_term_gru")

        self.fusion = layers.Dense(
            c.fusion_dim,
            activation="relu",
            kernel_regularizer=regularizers.l2(c.l2_fusion),
            name="user_representation",
        )
        self.output_dropout = layers.Dropout(c.dropout)
        self.predictor = layers.Dense(
            c.num_items,
            activation="softmax",
            kernel_regularizer=regularizers.l2(c.l2_output),
            name="next_item_distribution",
        )

    def _attribute_embeddings(self, leading_shape):
        ids = tf.range(self.config.num_attributes, dtype=tf.int32)
        emb = self.attribute_embedding(ids)  # [A,D]
        # Broadcast [A,D] to leading_shape + [A,D].
        target = tf.concat([leading_shape, tf.shape(emb)], axis=0)
        reshape = tf.concat([tf.ones_like(leading_shape), tf.shape(emb)], axis=0)
        return tf.broadcast_to(tf.reshape(emb, reshape), target)

    def call(self, inputs: Dict[str, tf.Tensor], training=None, return_attention=False):
        user_id = tf.cast(inputs["user_id"], tf.int32)
        last_item_id = tf.cast(inputs["last_item_id"], tf.int32)
        long_ids = tf.cast(inputs["long_attr_item_ids"], tf.int32)       # [B,L,A,N]
        long_w = tf.cast(inputs["long_attr_weight_class"], tf.int32)    # [B,L,A]
        short_ids = tf.cast(inputs["short_attr_item_ids"], tf.int32)    # [B,A,N]
        short_w = tf.cast(inputs["short_attr_weight_class"], tf.int32) # [B,A]
        short_seq = tf.cast(inputs["short_sequence_ids"], tf.int32)     # [B,T]

        user = self.user_embedding(user_id)             # [B,D]
        last_item = self.item_embedding(last_item_id)   # [B,D]

        # ----- Long-term intention learning -----
        long_item_mask = tf.not_equal(long_ids, 0)      # [B,L,A,N]
        long_attr_mask = tf.reduce_any(long_item_mask, axis=-1)  # [B,L,A]
        long_session_mask = tf.reduce_any(long_attr_mask, axis=-1)  # [B,L]
        long_item_emb = self.item_embedding(long_ids)   # [B,L,A,N,D]
        long_weight_emb = self.weight_embedding(long_w) # [B,L,A,D]

        b = tf.shape(long_ids)[0]
        l = tf.shape(long_ids)[1]
        a = tf.shape(long_ids)[2]
        d = self.config.embedding_dim

        long_attr_emb = self._attribute_embeddings(tf.stack([b, l]))  # [B,L,A,D]
        long_user = tf.broadcast_to(user[:, tf.newaxis, :], [b, l, d])
        long_last = tf.broadcast_to(last_item[:, tf.newaxis, :], [b, l, d])

        long_intention, long_attn = self.iau(
            user_emb=long_user,
            last_item_emb=long_last,
            attr_item_embs=long_item_emb,
            attribute_embs=long_attr_emb,
            weight_embs=long_weight_emb,
            item_mask=long_item_mask,
            attribute_mask=long_attr_mask,
            training=training,
        )  # [B,L,D]

        encoded_history = self.long_encoder(
            long_intention, mask=long_session_mask, training=training
        )  # [B,L,2G]

        # ----- Current / short-term intention -----
        short_item_mask = tf.not_equal(short_ids, 0)      # [B,A,N]
        short_attr_mask = tf.reduce_any(short_item_mask, axis=-1)  # [B,A]
        short_item_emb = self.item_embedding(short_ids)   # [B,A,N,D]
        short_weight_emb = self.weight_embedding(short_w) # [B,A,D]
        short_attr_emb = self._attribute_embeddings(tf.stack([b])) # [B,A,D]

        current_intention, short_attn = self.iau(
            user_emb=user,
            last_item_emb=last_item,
            attr_item_embs=short_item_emb,
            attribute_embs=short_attr_emb,
            weight_embs=short_weight_emb,
            item_mask=short_item_mask,
            attribute_mask=short_attr_mask,
            training=training,
        )  # [B,D]

        # Eq. (12)-(14): current intention conditions long-term session attention.
        long_short, eta = self.temporal_attention(
            encoded_history,
            current_intention,
            session_mask=long_session_mask,
            training=training,
        )  # [B,2G]

        # Eq. (15)-(18): sequential pattern in the current session.
        short_seq_mask = tf.not_equal(short_seq, 0)
        short_seq_emb = self.item_embedding(short_seq)
        ht = self.short_encoder(short_seq_emb, mask=short_seq_mask, training=training)  # [B,G]

        # Eq. (19): U = [I ; I^(long*short) ; h_t] then dense abstraction.
        combined = tf.concat([current_intention, long_short, ht], axis=-1)
        user_repr = self.fusion(combined)

        # Eq. (20): probability distribution over next items.
        logits_prob = self.predictor(self.output_dropout(user_repr, training=training))

        if return_attention:
            return logits_prob, {
                "long_item_attention": long_attn["item_attention"],
                "long_attribute_attention": long_attn["attribute_attention"],
                "short_item_attention": short_attn["item_attention"],
                "short_attribute_attention": short_attn["attribute_attention"],
                "temporal_attention": eta,
            }
        return logits_prob

    def get_config(self):
        return {"config": asdict(self.config)}


# -----------------------------------------------------------------------------
# Training / evaluation helpers
# -----------------------------------------------------------------------------

def make_optimizer(learning_rate: float = 0.01):
    """Paper setting: Adam, learning rate 0.01."""
    return keras.optimizers.Adam(learning_rate=learning_rate)


def make_loss():
    """Cross-entropy for one-hot next-item targets.

    The paper states that y is one-hot and y_hat is a probability distribution,
    and prints Eq. (21) with both positive and negative BCE-style terms.
    For a single mutually-exclusive next item with a softmax output, categorical
    cross-entropy is the standard direct implementation of -sum y_i log(y_hat_i).
    If exact reproduction of the printed Eq. (21) is required, replace this with
    BinaryCrossentropy() and document that experimental choice explicitly.
    """
    return keras.losses.CategoricalCrossentropy()


def recall_at_k(y_true, y_pred, k=20):
    true_idx = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    topk = tf.math.top_k(y_pred, k=k).indices
    hit = tf.reduce_any(tf.equal(topk, true_idx[:, None]), axis=-1)
    return tf.reduce_mean(tf.cast(hit, tf.float32))


def ndcg_at_k(y_true, y_pred, k=20):
    true_idx = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    topk = tf.math.top_k(y_pred, k=k).indices
    match = tf.equal(topk, true_idx[:, None])
    found = tf.reduce_any(match, axis=-1)
    rank = tf.argmax(tf.cast(match, tf.int32), axis=-1, output_type=tf.int32) + 1
    gain = 1.0 / (tf.math.log(tf.cast(rank + 1, tf.float32)) / tf.math.log(2.0))
    return tf.reduce_mean(tf.where(found, gain, 0.0))


def mrr_at_k(y_true, y_pred, k=20):
    true_idx = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    topk = tf.math.top_k(y_pred, k=k).indices
    match = tf.equal(topk, true_idx[:, None])
    found = tf.reduce_any(match, axis=-1)
    rank = tf.argmax(tf.cast(match, tf.int32), axis=-1, output_type=tf.int32) + 1
    rr = 1.0 / tf.cast(rank, tf.float32)
    return tf.reduce_mean(tf.where(found, rr, 0.0))


# -----------------------------------------------------------------------------
# Model inspection helpers for GitHub / dissertation reporting
# -----------------------------------------------------------------------------

def dummy_inputs(
    batch_size: int,
    long_sessions: int,
    num_attributes: int,
    items_per_attribute: int,
    short_sequence_length: int,
):
    """Create shape-correct tensors to build/profile the model."""
    return {
        "user_id": tf.ones([batch_size], tf.int32),
        "last_item_id": tf.ones([batch_size], tf.int32),
        "long_attr_item_ids": tf.ones(
            [batch_size, long_sessions, num_attributes, items_per_attribute], tf.int32
        ),
        "long_attr_weight_class": tf.zeros(
            [batch_size, long_sessions, num_attributes], tf.int32
        ),
        "short_attr_item_ids": tf.ones(
            [batch_size, num_attributes, items_per_attribute], tf.int32
        ),
        "short_attr_weight_class": tf.zeros(
            [batch_size, num_attributes], tf.int32
        ),
        "short_sequence_ids": tf.ones(
            [batch_size, short_sequence_length], tf.int32
        ),
    }


def parameter_report(model: keras.Model) -> Dict[str, int]:
    trainable = sum(int(tf.size(v)) for v in model.trainable_variables)
    non_trainable = sum(int(tf.size(v)) for v in model.non_trainable_variables)
    return {
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
        "total_parameters": trainable + non_trainable,
    }


def profile_flops(model: keras.Model, example_inputs: Dict[str, tf.Tensor]) -> Optional[int]:
    """Return TensorFlow profiler FLOPs for one forward pass when available.

    FLOP counting is implementation/backend dependent; report the TensorFlow
    version, hardware, batch size, and input shapes alongside this number.
    """
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        from tensorflow.python.profiler.model_analyzer import profile
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

        specs = {
            k: tf.TensorSpec(v.shape, v.dtype, name=k) for k, v in example_inputs.items()
        }

        @tf.function
        def forward(**kwargs):
            return model(kwargs, training=False)

        concrete = forward.get_concrete_function(**specs)
        frozen = convert_variables_to_constants_v2(concrete)
        graph = frozen.graph
        opts = ProfileOptionBuilder.float_operation()
        result = profile(graph=graph, options=opts)
        return int(result.total_float_ops) if result is not None else None
    except Exception as exc:  # profiler availability differs across TF versions
        print(f"FLOP profiling unavailable: {exc}")
        return None


def theoretical_complexity_symbols():
    """Symbolic leading-order terms for reporting (forward pass, per batch item).

    Symbols:
      L: long-term sessions
      A: attribute types
      N: items per attribute group
      T: current-session length
      D: embedding dimension
      H: attention heads
      R: GRU hidden size
      V: number of candidate items

    Notes:
      - Item attention is linear in the number of attended items (no N^2 term).
      - Collaborative projection adds O(A*H^2) per IAU application.
      - Dense full-catalog prediction contributes O(V*fusion_dim).
    """
    return {
        "item_attention": "O((L+1) * A * N * D^2)",
        "collaborative_attribute_attention": "O((L+1) * (A*D*attention_dim + A*H^2 + A*H*D))",
        "long_term_BiGRU": "O(L * R * (D + R))",
        "temporal_attention": "O(L * D * R)",
        "short_term_GRU": "O(T * R * (D + R))",
        "fusion": "O((D + 3R) * fusion_dim)",
        "full_catalog_prediction": "O(V * fusion_dim)",
    }


if __name__ == "__main__":
    # Example configuration. Replace dataset sizes with the processed dataset.
    cfg = DIARecConfig(
        num_users=943,
        num_items=1682,
        num_attributes=7,
        embedding_dim=16,
        num_heads=3,
        attention_dim=30,
        gru_units=16,
        fusion_dim=30,
        dropout=0.5,
    )

    model = DIARec(cfg, name="DIARec")
    x = dummy_inputs(
        batch_size=2,
        long_sessions=5,
        num_attributes=cfg.num_attributes,
        items_per_attribute=10,
        short_sequence_length=20,
    )
    y = model(x, training=False)

    model.summary()
    print("Output shape:", y.shape)
    print("Parameter report:", parameter_report(model))
    print("Theoretical complexity:", theoretical_complexity_symbols())
    print("FLOPs (batch=2):", profile_flops(model, x))
