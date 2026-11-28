// A sigmoid + bias MoE route at any expert count: picks, weights, expert groups, the ordered combine.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0 // planted faults: 1 ties to the higher id, 2 normalising flipped, 3 combine skips the shared slot
#endif

struct TfMoeArgs {
  uint rows, experts, topk, shared; // shared 1: slot topk is expert `experts` at weight 1
  uint ldy; // y's row stride (the hidden width)
  uint score, normalize, groups, keep; // score 1: rank sigmoid(logit) + bias, else logit + bias; groups of experts
  float scale; // routed weights times this (after normalising)
};

inline float tf_moe_sigmoid(float x) {
  const float e = precise::exp(fabs(x));
  const float y = 1.0f / (1.0f + e);
  return x < 0.0f ? y : 1.0f - y;
}

inline float tf_moe_choice(const device float* lg, const device float* bias, uint e, uint score) {
  return (score != 0 ? tf_moe_sigmoid(lg[e]) : lg[e]) + bias[e];
}

// The `keep` groups (of experts / groups, at least two each) of the best top-2 key sums, ties to the lower group.
inline uint tf_moe_kept(const device float* key, constant TfMoeArgs& a) {
  if (a.groups <= 1) return ~0u;
  const uint gs = a.experts / a.groups;
  float sums[32];
  for (uint g = 0; g < a.groups; ++g) {
    float x = -INFINITY, y = -INFINITY;
    for (uint e = g * gs; e < g * gs + gs; ++e) {
      const float c = key[e];
      if (c > x) { y = x; x = c; } else if (c > y) { y = c; }
    }
    sums[g] = x + y;
  }
  uint kept = 0u;
  for (uint k = 0; k < a.keep; ++k) {
    uint best = 32u;
    for (uint g = 0; g < a.groups; ++g)
      if (!((kept >> g) & 1u) && (best == 32u || sums[g] > sums[best])) best = g;
    kept |= 1u << best;
  }
  return kept;
}

// Row r's top-k by fp32 rank key (computed once into keys[r]) in the kept groups, lower id on ties; sigmoid weights.
[[kernel, max_total_threads_per_threadgroup(64)]] void tf_moe_select(const device float* logits [[buffer(0)]], const device float* bias [[buffer(1)]],
    constant TfMoeArgs& a [[buffer(2)]], device uint* expert_of [[buffer(3)]], device float* weight [[buffer(4)]],
    device float* keys [[buffer(5)]], uint r [[thread_position_in_grid]]) {
  if (r >= a.rows) return;
  const uint slots = a.topk + a.shared;
  const device float* lg = logits + (size_t)r * a.experts;
  device float* key = keys + (size_t)r * a.experts;
  device uint* pick = expert_of + (size_t)r * slots;
  device float* w = weight + (size_t)r * slots;
  for (uint e = 0; e < a.experts; ++e) key[e] = tf_moe_choice(lg, bias, e, a.score);
  const uint kept = tf_moe_kept(key, a), gs = a.groups <= 1 ? a.experts : a.experts / a.groups;
  float total = 0.0f;
  for (uint s = 0; s < a.topk; ++s) {
    uint best = a.experts;
    float top = -INFINITY;
    for (uint e = 0; e < a.experts; ++e) {
      bool taken = !((kept >> (e / gs)) & 1u);
      for (uint t = 0; t < s; ++t) taken = taken || pick[t] == e;
      const float v = key[e];
      if (!taken && (best == a.experts || (TF_CTL == 1 ? v >= top : v > top))) { best = e; top = v; }
    }
    pick[s] = best;
    w[s] = tf_moe_sigmoid(lg[best]);
    total += w[s];
  }
  const bool norm = TF_CTL == 2 ? a.normalize == 0 : a.normalize != 0;
  for (uint s = 0; s < a.topk; ++s) w[s] = (norm ? w[s] / (total + 1e-20f) : w[s]) * a.scale;
  if (a.shared != 0) { pick[a.topk] = a.experts; w[a.topk] = 1.0f; }
}

struct TfMoeItem {
  uint expert, first, count;
};

// Pairs grouped by expert, in pair order within one, then items of at most 16 pairs: one thread, any count.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_moe_plan(const device uint* expert_of [[buffer(0)]], constant TfMoeArgs& a [[buffer(1)]],
    device uint* bucket [[buffer(2)]], device uint* members [[buffer(3)]], device TfMoeItem* items [[buffer(4)]],
    device uint* count [[buffer(5)]], uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  const uint pairs = a.rows * (a.topk + a.shared), groups = a.experts + a.shared;
  for (uint e = 0; e < groups; ++e) bucket[e] = 0;
  for (uint p = 0; p < pairs; ++p) bucket[expert_of[p]] += 1;
  uint at = 0, n = 0;
  for (uint e = 0; e < groups; ++e) {
    const uint c = bucket[e];
    for (uint i = 0; i < c; i += 16) items[n++] = TfMoeItem{e, at + i, min(16u, c - i)};
    bucket[e] = at;
    at += c;
  }
  for (uint p = 0; p < pairs; ++p) members[bucket[expert_of[p]]++] = p;
  count[0] = n;
}

// out[r][d] = bf16 of each slot's weight times its fp32 y, added in slot order.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_moe_combine(const device float* y [[buffer(0)]], const device float* weight [[buffer(1)]],
    constant TfMoeArgs& a [[buffer(2)]], device bfloat* out [[buffer(3)]], uint2 id [[thread_position_in_grid]]) {
  const uint d = id.x, r = id.y;
  if (d >= a.ldy || r >= a.rows) return;
  const uint slots = a.topk + a.shared, used = TF_CTL == 3 ? a.topk : slots;
  float acc = 0.0f;
  for (uint s = 0; s < used; ++s) {
    const size_t p = (size_t)r * slots + s;
    acc = fma(weight[p], y[p * a.ldy + d], acc);
  }
  out[(size_t)r * a.ldy + d] = bfloat(acc);
}
