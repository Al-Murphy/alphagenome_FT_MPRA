from alphagenome_ft_mpra import load_oracle
import jax

oracle = load_oracle(
    "/grid/koo/home/shared/alphagenome_encoder/mpra-HepG2-optimal/stage2/",
    # Optional construct pieces (set to None to skip)
    left_adapter=None,
    right_adapter=None,
    promoter="TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG",
    barcode="AGAGACTGAGGCCAC"
)

# mode="core": add left/right adapters + promoter + barcode (if provided)
# mode="flanked": add promoter + barcode (if provided)
# mode="full": no sequence additions

# Usage 1) onehot in shape: (S, 4) or (B, S, 4)
#scores = oracle.predict(onehot, mode="core")

# Usage 2) string convenience wrapper
scores = oracle.predict_sequences(["AGGACCGGATCAACTCCTAACCCTAACCCTAACCCTAACCCTAACCCCTAACCCTAACCCTAACCCTAACCCTCGCGGTACCCTCAGCCGGCCCGCCCGCCCGGGTCTGACCTGAGGAGAACTGTGCTCCGCCTTCAGAGTACCACCGAAATCTGTGCAGAGGACAACGCAGCTCCGCCCTCGCGGTGCTCTCCGGGTCTGTGCTGAGGAGAACGCATTGCGTGAACCGA"], mode="core")
print(scores)