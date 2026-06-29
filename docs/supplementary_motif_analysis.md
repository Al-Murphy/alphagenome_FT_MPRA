# Simplified motif-analysis text

## Figure legend

Supplementary Fig. X. Cross-species fine-tuning uses conserved partial motifs for developmental activity but learns housekeeping motifs de novo. We compared the frozen pretrained encoder, a probing model, and a fine-tuned Drosophila DeepSTARR model. Developmental motifs were recovered mainly from partial motifs already present in the frozen encoder, whereas housekeeping motifs were largely acquired during fine-tuning.

## Results

Motif-level analysis of the fine-tuned Drosophila encoder showed that developmental enhancer activity was supported largely by cis-regulatory motifs already captured by the human- and mouse-pretrained encoder, whereas housekeeping activity relied more on motifs learned de novo during fine-tuning. This was evident in the recovery of developmental factors such as AP-1/Jra and SREBP from conserved partial motifs, versus insect-specific housekeeping motifs such as Dref and M1BP that were not present in the frozen encoder.

## Methods

To characterize how encoder-only adaptation bridges the human/mouse-to-Drosophila species shift, we compared the frozen pretrained encoder, a probing model, and a fine-tuned model on the Drosophila DeepSTARR developmental and housekeeping tracks. First-layer convolutional filters were interpreted as partial motifs from their activations on test sequences and matched to JASPAR with TOMTOM. Whole motifs were then identified from gradient-based attributions with TF-MoDISco and matched to JASPAR. A whole motif recovered by fine-tuning but not probing was classified as conserved if the frozen encoder already contained a matching first-layer filter and as novel otherwise.
