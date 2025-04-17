#import "@preview/ilm:1.4.1": *

#set text(lang: "en")

#show: ilm.with(
  title: [CS 336: Assignment 1],
  author: "Brandon Snider",
  date: datetime(year: 2025, month: 04, day: 15),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
)

#set enum(numbering: "a)")
#set heading(numbering: none)

= BPE Tokenizer

== Problem (`unicode1`): Understanding Unicode (1 point)

+ `chr(0)` returns `'\x00'`

+ The string representation (`__repr__()`) shows the escape sequence (`'\x00'`), while the printed representation shows no visible output.

+ In the printed representation (within a call to `print`), this character produces no visible output; in a string representation (the first and third examples), the escape sequence `\x00` appears in the ouptut.

== Problem (`unicode2`): Unicode Encodings (3 points)

+ By starting from a tiny initial vocabulary (256 byte values) and learning efficient merges based on data, we produce a compact, efficient vocabulary. We can then fit more semantic content into a fixed-size context window.

+ The function treats each byte as a Unicode character, but UTF-8 represents characters as sequences of 1-4 bytes. The string `"¢"` would be decoded incorrectly, because the single character's UTF-8 representation is 2 bytes.

+ `0x80 0x80`, because `0x80` (equivalently `10000000`) is a continuation byte, and cannot be used as the first byte in a unicode sequence.

== Problem (`train_bpe_tinystories`): BPE Training on TinyStories (2 points)

+ Time: 135.32s (0.038h) \
  Memory: 4GB (per Scalene) \
  Longest token: `' accomplishment'`. This makes sense. With a fairly large vocabulary and a dataset of clean English text, one would expect the longest tokens to be long strings of valid English that appear contiguously in the dataset.

+ Pre-tokenization took roughly half of the overall training time (102s). The specific bottleneck is creating a bytes object for each individual character in each regex match to construct the keys in the table of coarse-grained tokens.

== Problem (`train_bpe_expts_owt`): BPE Training on OpenWebText (2 points)

+ \@TODO
+ \@TODO

== Problem (`tokenizer_experiments`): Experiments with Tokenizers (4 points)

+ \@TODO
+ \@TODO
+ \@TODO
+ \@TODO

= Transformer Language Model Architecture

== Problem (`transformer_accounting`): LM resource accounting (5 points)

+ Expression for the total parameter count:
  $ p_"total" = p_"embedding" + p_"layers" + p_"ln_final" + p_"lm_head"  $
  Parameter count of the embedding matrix:
  $ p_"embedding" = "vocab_size" times d_"model" = 50,257 times 1,600 = 80,411,200 $
  Parameter count of all transformer blocks:
  $ p_"layers" &= "num_layers" times p_"layer" \
    p_"layer" &= (2 times p_"ln") + p_"attn" + p_"ffn" \ 
    &= (2 times d_"model") + (p_"wqkv" + p_"out_proj") + (p_"w1" + p_"w2" + p_"w3") \
    &= (2 times d_"model") + (d_"model" times 3 times d_"model" + d_"model" times d_"model") + (3 times d_"model" times d_"ff") \
    &= (2 times 1,600) + (1,600 times 3 times 1,600 + 1,600 times 1,600) + (3 times 1,600 times 6,400) \
    &= 40,963,200 \
    p_"layers" &= 48 times 40,963,200 \
    &= 1,966,233,600
  $
  Parameter count of the final layer norm:
  $ p_"ln" = d_"model" = 1600 $
  Parameter count of the LM head:
  $ p_"lm_head" = d_"model" times "vocab_size" = 1,600 times 50,257 = 80,411,200 $
  Final parameter count:
  $ p_"total" &= p_"embedding" + p_"layers" + p_"ln_final" + p_"lm_head" \
    &= 80,411,200 + 1,966,233,600 + 1600 + 80,411,200 \
    &= 2,127,057,600
  $
  Assuming each parameter is represented using single-recision floating point (4 bytes), the memory required to load the model is:
  $ "memory" &= p_"total" times "memory_per_param" \
    &= 2,127,057,600 times 4 \
    &= 8,508,230,400 "bytes" \
    &approx 8.51"GB"
  $ 
  
+ *GPT-2 XL analysis:*

  FLOPs to compute queries, keys, and values across all layers (using a $W_"qkv"$ matrix):
  $ F_"attn_qkv" &= "num_layers" times [2 times d_"model" times (3 times d_"model") times "context_length"] \
    &= 48 times [2 times 1,600 times (3 times 1600) times 1,024] \
    &= 754,974,720,000 $
  
  Computing attention weights ($Q^T K$) across all layers:
  $ F_"attn_weights" &= "num_layers" times (2 times "context_length" times d_"model" times "context_length") \
    &= 48 times (2 times 1,024 times 1,600 times 1,024) \
    &= 161,061,273,600 $

  Computing attention values ($W V$, where $W$ represents normalized attention weights):
  $ F_"attn_values" &= "num_layers" times (2 times "context_length" times "context_length" times d_"model") \
    &= 48 times (2 times 1,024 times 1,024 times 1,600) \
    &= 161,061,273,600 $

  Output projection after the attention operation, across all layers:
  $ F_"attn_out" &= "num_layers" times (2 times d_"model" times d_"model" times "context_length") \
    &= 48 times (2 times 1,600 times 1,600 times 1,024) \
    &= 251,658,240,000 $
  
  Up projection in the FFN, across all layers:
  $ F_"ffn_up" &= "num_layers" times (2 times d_"model" times d_"ff" times "context_length") \
    &= 48 times (2 times 1,600 times 6,400 times 1,024) \
    &= 1,006,632,960,000 $

  Parallel up projection in the FFN (identical):
  $ F_"ffn_parallel" = F_"ffn_up" = 1,006,632,960,000 $

  Down projection in the FFN, across all layers (same $m,n, p$; different order): 
  $ F_"ffn_down" = F_"ffn_up" = 1,006,632,960,000 $

  Final output projection (LM head):
  $ F_"lm_head" &= 2 times "vocab_size" times d_"model" times "context_length" \
    &= 2 times 50,257 times 1,600 times 1,024 \
    &= 164,682,137,600 $

  Total FLOPs:
  $ F_"total" &= F_"attn" + F_"ffn" + F_"lm_head" \ 

    F_"attn" &= F_"attn_qkv" + F_"attn_weights" + F_"attn_values" + F_"attn_out" \
      &= 754,974,720,000 + (2 times 161,061,273,600) + 251,658,240,000 \
      &= 1,328,755,507,200 \
      &approx 1.33 times 10^12 "FLOPs" \

    F_"ffn" &= F_"ffn_up" + F_"ffn_parallel" + F_"ffn_down" \
      &= 3 times 1,006,632,960,000 \
      &= 3,019,898,880,000 \
      &approx 3.02 times 10^12 "FLOPs" \

    F_"total" &= 1,328,755,507,200 + 3,019,898,880,000 + 164,682,137,600 \
      &= 4,513,336,524,800 \
      &approx 4.51 times 10^12 "FLOPs" $
  
  Proportions:
  $ 
    P_"attn_qkv" &= 754,974,720,000 " / " F_"total" approx 16.7% \
    P_"attn_weights" &= 161,061,273,600 " / " F_"total" approx 3.6% \
    P_"attn_values" &= 161,061,273,600 " / " F_"total" approx 3.6% \
    P_"attn_out" &= 251,658,240,000 " / " F_"total" approx 5.6% \
    P_"attn" &= 1,328,755,507,200 " / " F_"total" approx 29% \
    \
    P_"ffn_up" &= 1,006,632,960,000 " / " F_"total" approx 22.3% \
    P_"ffn_parallel" &= 1,006,632,960,000 " / " F_"total" approx 22.3% \
    P_"ffn_down" &= 1,006,632,960,000 " / " F_"total" approx 22.3% \
    P_"ffn" &= 3,019,898,880,000 " / " F_"total" approx 66.9% \
    \
    P_"lm_head" &= 164,682,137,600 " / " F_"total" approx 3.7% \
  $

+ The FFNs require the most FLOPs by far, accounting for roughly $67%$ of the total (with each of the three matrix multiplications in the FFNs contributing equally). The attention blocks are the next most significant, accounting for roughly $29%$ of the total.

+ *GPT-2 small analysis:*
  
  FLOPs to compute queries, keys, and values across all layers (using a $W_"qkv"$ matrix):
  $ F_"attn_qkv" &= "num_layers" times [2 times d_"model" times (3 times d_"model") times "context_length"] \
    &= 12 times [2 times 768 times (3 times 768) times 1,024] \
    &= 43,486,543,872 $

  Computing attention weights ($Q^T K$) across all layers:
  $ F_"attn_weights" &= "num_layers" times (2 times "context_length" times d_"model" times "context_length") \
    &= 12 times (2 times 1,024 times 768 times 1,024) \
    &= 19,327,352,832 $

  Computing attention values ($W V$, where $W$ represents normalized attention weights):
  $ F_"attn_values" &= "num_layers" times (2 times "context_length" times "context_length" times d_"model") \
    &= 12 times (2 times 1,024 times 1,024 times 768) \
    &= 19,327,352,832 $

  Output projection after the attention operation, across all layers:
  $ F_"attn_out" &= "num_layers" times (2 times d_"model" times d_"model" times "context_length") \
    &= 12 times (2 times 768 times 768 times 1,024) \
    &= 14,495,514,624 $
  
  Up projection in the FFN, across all layers:
  $ F_"ffn_up" &= "num_layers" times (2 times d_"model" times d_"ff" times "context_length") \
    &= 12 times (2 times 768 times 3,072 times 1,024) \
    &= 57,982,058,496 $

  Parallel up projection in the FFN (identical):
  $ F_"ffn_parallel" = F_"ffn_up" = 57,982,058,496 $

  Down projection in the FFN, across all layers (same $m,n,p$; different order):
  $ F_"ffn_down" = F_"ffn_up" = 57,982,058,496 $

  Final output projection (LM head):
  $ F_"lm_head" &= 2 times "vocab_size" times d_"model" times "context_length" \
    &= 2 times 50,257 times 768 times 1,024 \
    &= 79,047,426,048 $

  Total FLOPs:
  $ F_"attn" &= F_"attn_qkv" + F_"attn_weights" + F_"attn_values" + F_"attn_out" \
    &= 43,486,543,872 + 19,327,352,832 + 19,327,352,832 + 14,495,514,624 \
    &= 96,636,764,160 \

    F_"ffn" &= F_"ffn_up" + F_"ffn_parallel" + F_"ffn_down" \
    &= 57,982,058,496 + 57,982,058,496 + 57,982,058,496 \
    &= 173,946,175,488 \

    F_"total" &= F_"attn" + F_"ffn" + F_"lm_head" \
    &= 96,636,764,160 + 173,946,175,488 + 79,047,426,048 \
    &= 349,630,365,696 $

  Proportions:

  $ 
  P_"attn_qkv" &= 43,486,543,872 " / " F_"total" approx 12.4% \
  P_"attn_weights" &= 19,327,352,832 " / " F_"total" approx 5.5% \
  P_"attn_values" &= 19,327,352,832 " / " F_"total" approx 5.5% \
  P_"attn_out" &= 14,495,514,624 " / " F_"total" approx 4.1% \
  P_"attn" &= 96,636,764,160 " / " F_"total" approx 27.6% \
  \
  P_"ffn_up" &= 57,982,058,496 " / " F_"total" approx 16.6% \
  P_"ffn_parallel" &= 57,982,058,496 " / " F_"total" approx 16.6% \
  P_"ffn_down" &= 57,982,058,496 " / " F_"total" approx 16.6% \
  P_"ffn" &= 173,946,175,488 " / " F_"total" approx 49.8% \
  \
  P_"lm_head" &= 79,047,426,048 " / " F_"total" approx 22.6% \
$

  *GPT-2 medium analysis:*

  FLOPs to compute queries, keys, and values across all layers (using a $W_"qkv"$ matrix):
  $ F_"attn_qkv" &= "num_layers" times [2 times d_"model" times (3 times d_"model") times "context_length"] \
    &= 24 times [2 times 1,024 times (3 times 1,024) times 1,024] \
    &= 154,618,822,656 $

  Computing attention weights ($Q^T K$) across all layers:
  $ F_"attn_weights" &= "num_layers" times (2 times "context_length" times d_"model" times "context_length") \
    &= 24 times (2 times 1,024 times 1,024 times 1,024) \
    &= 51,539,607,552 $

  Computing attention values ($W V$, where $W$ represents normalized attention weights):
  $ F_"attn_values" &= "num_layers" times (2 times "context_length" times "context_length" times d_"model") \
    &= 24 times (2 times 1,024 times 1,024 times 1,024) \
    &= 51,539,607,552 $

  Output projection after the attention operation, across all layers:
  $ F_"attn_out" &= "num_layers" times (2 times d_"model" times d_"model" times "context_length") \
    &= 24 times (2 times 1,024 times 1,024 times 1,024) \
    &= 51,539,607,552 $
  
  Up projection in the FFN, across all layers:
  $ F_"ffn_up" &= "num_layers" times (2 times d_"model" times d_"ff" times "context_length") \
    &= 24 times (2 times 1,024 times 4,096 times 1,024) \
    &= 206,158,430,208 $

  Parallel up projection in the FFN (identical):
  $ F_"ffn_parallel" = F_"ffn_up" = 206,158,430,208 $

  Down projection in the FFN, across all layers (same $m,n,p$; different order):
  $ F_"ffn_down" = F_"ffn_up" = 206,158,430,208 $

  Final output projection (LM head):
  $ F_"lm_head" &= 2 times "vocab_size" times d_"model" times "context_length" \
    &= 2 times 50,257 times 1,024 times 1,024 \
    &= 105,396,568,064 $

  Total FLOPs:
  $ F_"attn" &= F_"attn_qkv" + F_"attn_weights" + F_"attn_values" + F_"attn_out" \
    &= 154,618,822,656 + 51,539,607,552 + 51,539,607,552 + 51,539,607,552 \
    &= 309,237,645,312 \

    F_"ffn" &= F_"ffn_up" + F_"ffn_parallel" + F_"ffn_down" \
    &= 206,158,430,208 + 206,158,430,208 + 206,158,430,208 \
    &= 618,475,290,624 \

    F_"total" &= F_"attn" + F_"ffn" + F_"lm_head" \
    &= 309,237,645,312 + 618,475,290,624 + 105,396,568,064 \
    &= 1,033,109,504,000 $

  Proportions:

  $ 
    P_"attn_qkv" &= 154,618,822,656 " / " F_"total" approx 15.0% \
    P_"attn_weights" &= 51,539,607,552 " / " F_"total" approx 5.0% \
    P_"attn_values" &= 51,539,607,552 " / " F_"total" approx 5.0% \
    P_"attn_out" &= 51,539,607,552 " / " F_"total" approx 5.0% \
    P_"attn" &= 309,237,645,312 " / " F_"total" approx 29.9% \
    \
    P_"ffn_up" &= 206,158,430,208 " / " F_"total" approx 20.0% \
    P_"ffn_parallel" &= 206,158,430,208 " / " F_"total" approx 20.0% \
    P_"ffn_down" &= 206,158,430,208 " / " F_"total" approx 20.0% \
    P_"ffn" &= 618,475,290,624 " / " F_"total" approx 59.9% \
    \
    P_"lm_head" &= 105,396,568,064 " / " F_"total" approx 10.2% \
  $

  *GPT-2 large analysis:*

  FLOPs to compute queries, keys, and values across all layers (using a $W_"qkv"$ matrix):
  $ F_"attn_qkv" &= "num_layers" times [2 times d_"model" times (3 times d_"model") times "context_length"] \
    &= 36 times [2 times 1,280 times (3 times 1,280) times 1,024] \
    &= 362,387,865,600 $

  Computing attention weights ($Q^T K$) across all layers:
  $ F_"attn_weights" &= "num_layers" times (2 times "context_length" times d_"model" times "context_length") \
    &= 36 times (2 times 1,024 times 1,280 times 1,024) \
    &= 96,636,764,160 $

  Computing attention values ($W V$, where $W$ represents normalized attention weights):
  $ F_"attn_values" &= "num_layers" times (2 times "context_length" times "context_length" times d_"model") \
    &= 36 times (2 times 1,024 times 1,024 times 1,280) \
    &= 96,636,764,160 $

  Output projection after the attention operation, across all layers:
  $ F_"attn_out" &= "num_layers" times (2 times d_"model" times d_"model" times "context_length") \
    &= 36 times (2 times 1,280 times 1,280 times 1,024) \
    &= 120,795,955,200 $
  
  Up projection in the FFN, across all layers:
  $ F_"ffn_up" &= "num_layers" times (2 times d_"model" times d_"ff" times "context_length") \
    &= 36 times (2 times 1,280 times 5,120 times 1,024) \
    &= 483,183,820,800 $

  Parallel up projection in the FFN (identical):
  $ F_"ffn_parallel" = F_"ffn_up" = 483,183,820,800 $

  Down projection in the FFN, across all layers (same $m,n,p$; different order):
  $ F_"ffn_down" = F_"ffn_up" = 483,183,820,800 $

  Final output projection (LM head):
  $ F_"lm_head" &= 2 times "vocab_size" times d_"model" times "context_length" \
    &= 2 times 50,257 times 1,280 times 1,024 \
    &= 131,745,710,080 $

  Total FLOPs:
  $ F_"attn" &= F_"attn_qkv" + F_"attn_weights" + F_"attn_values" + F_"attn_out" \
    &= 362,387,865,600 + 96,636,764,160 + 96,636,764,160 + 120,795,955,200 \
    &= 676,457,349,120 \

    F_"ffn" &= F_"ffn_up" + F_"ffn_parallel" + F_"ffn_down" \
    &= 483,183,820,800 + 483,183,820,800 + 483,183,820,800 \
    &= 1,449,551,462,400 \

    F_"total" &= F_"attn" + F_"ffn" + F_"lm_head" \
    &= 676,457,349,120 + 1,449,551,462,400 + 131,745,710,080 \
    &= 2,257,754,521,600 $

  Proportions:
  $ 
    P_"attn_qkv" &= 362,387,865,600 " / " F_"total" approx 16.1% \
    P_"attn_weights" &= 96,636,764,160 " / " F_"total" approx 4.3% \
    P_"attn_values" &= 96,636,764,160 " / " F_"total" approx 4.3% \
    P_"attn_out" &= 120,795,955,200 " / " F_"total" approx 5.4% \
    P_"attn" &= 676,457,349,120 " / " F_"total" approx 30.0% \
    \
    P_"ffn_up" &= 483,183,820,800 " / " F_"total" approx 21.4% \
    P_"ffn_parallel" &= 483,183,820,800 " / " F_"total" approx 21.4% \
    P_"ffn_down" &= 483,183,820,800 " / " F_"total" approx 21.4% \
    P_"ffn" &= 1,449,551,462,400 " / " F_"total" approx 64.2% \
    \
    P_"lm_head" &= 131,745,710,080 " / " F_"total" approx 5.8% \
  $

  Analysis:

  The FFN computations increasingly dominate as model size increases. The contribution from the LM head is significant (greater than the contribution from attention) at the smallest model size, and diminishes quickly as model size increases.
  
+ The total FLOPs required increases from $4,513,336,524,800$ to $149,522,795,724,800$. The FLOPs for all operations except the attention operation increase linearly in the length of the context window (by a factor of $2^4$, in this case). The FLOPs for the attention operation (both $Q^T K$ and $W V$, where $W$ represents the normalized attention weights) increase quadratically in the length of the context window (by a factor of $2^8$, in this case).