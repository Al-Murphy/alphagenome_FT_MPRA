"""
Sequence loader for extracting DNA sequences from reference genomes (hg19/hg38).

Adapted from clg/src/dataset_utils.py for use in alphagenome_FT_MPRA project.
"""

import os
import gzip
import urllib.request
import pysam


def get_genome(build, lcl_path='./.cache', Force=False):
    """
    Downloads genome build as fasta file from UCSC, 
    if it is not already downloaded. 
    
    Parameters:
    - build: str, genome build, must be one of ['hg19','hg38']
    - lcl_path: str, optional, the local path to save the downloaded files. Default is './.cache'
    - Force: bool, optional, whether to force download the genome. Default is False.
    
    Returns:
    - gen_pth: str, path to genome fasta file
    """
    #force build to lower case
    build = build.lower()
    #ensure correct specification
    assert build in ['hg19','hg38'], "build must be one of ['hg19','hg38']"
    #download genome if not already downloaded
    gen_pth = lcl_path+"/"+build+".fa"
    
    #check if cache folder exists
    if not os.path.exists(lcl_path):
        os.makedirs(lcl_path)

    if (not os.path.exists(gen_pth)) or Force:
        if build == 'hg19':
            url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
        elif build == 'hg38':
            url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
        print(f"Downloading {build} genome...")
        urllib.request.urlretrieve(url, gen_pth+'.gz')
        # Unzip the gz file
        with gzip.open(gen_pth+'.gz', 'rb') as f_in:
            with open(gen_pth, 'wb') as f_out:
                f_out.write(f_in.read())
    return gen_pth


class seq_loader(object):
    """
    Class object to generate dna sequence for a genomic deep learning model.
    
    Parameters:
    - build: str, the genome build to use, "hg38" or "hg19".
    - model_receptive_field: int, the receptive field of the model
    - alphabet: list, optional, the alphabet to use for the one-hot encoding. Default is ['A', 'C', 'G', 'T'].
    """
    def __init__(self, build, model_receptive_field, alphabet=['A', 'C', 'G', 'T']):
        #force build to lower case
        build = build.lower()
        
        #ensure correct specification
        assert build in ['hg19','hg38'], "build must be one of ['hg19','hg38']"
        #make sure model's receptive field is specified
        assert model_receptive_field>0, "model_receptive_field must be greater than 0"
    
        #Get the genome data
        gen_path = get_genome(build)
        self.genome_dat = pysam.Fastafile(gen_path)
        self.model_receptive_field = model_receptive_field
        #check if any modulus of 2
        self.mod = model_receptive_field % 2
        self.alphabet = alphabet
    
    def get_chr_len(self, chrom):
        return(self.genome_dat.get_reference_length(chrom))
        
    def get_seq_start(self, chrom, seq_start, strand,
                ohe=True, rev_comp=False, pad_seq=True):
        """
        Get the DNA sequence for a given chromosome, start and end position.
        
        Parameters:
        - chrom: str, the chromosome to get the sequence from
        - seq_start: int, the start position of the sequence of interest.
        - ohe: bool, whether the input is one-hot encoded or not. Default = True.
        - rev_comp: bool, whether to get the reverse complement of the sequence. Note this 
        will not affect the strand of the gene as the returned sequence will be the reversed
        and compliment DNA returned for negative strand anyway.
        - pad_seq: bool, whether to pad the sequence with N's if the sequence is smaller than
        the model's receptive field (when TSS is at start or end of chrom). Default = True.
        
        Returns:
        - seq: str or torch.tensor, shape=(len(alphabet), seq_len), the DNA sequence
        """
        
        #work out the start and end positions
        start = max(0, seq_start)
        #if padding needed because start is at the beginning of the chromosome, get amount
        pad_N_strt = max(seq_start*-1, 0)
        #max end is the chromosome length
        chrom_len = self.genome_dat.get_reference_length(chrom)
        end = min(seq_start+self.model_receptive_field+self.mod, chrom_len)
        pad_N_end = max((seq_start + self.model_receptive_field+self.mod - chrom_len), 0)
        #get the sequence (with padding if necessary)
        if pad_seq:
            seq = ("N"*pad_N_strt)+self.genome_dat.fetch(chrom, start, end).upper()+("N"*pad_N_end)
        else: #no padding
            seq = self.genome_dat.fetch(chrom, start, end).upper()
        #convert to tensor if needed    
        if ohe:
            from tangermeme.utils import one_hot_encode
            seq = one_hot_encode(seq, force=True).unsqueeze(0)
        #get strand specific gene sequence
        if strand == "-":
            seq = self._reverse_complement_dna(seq, ohe=ohe)
            
        if rev_comp:
            seq = self._reverse_complement_dna(seq, ohe=ohe)
        
        return seq
    
    def get_seq(self, chrom, gene_start, strand,
                ohe=True, rev_comp=False, pad_seq=True):
        """
        Get the DNA sequence for a given chromosome, start and end position.
        
        Parameters:
        - chrom: str, the chromosome to get the sequence from
        - gene_start: int, the start position of the TSS for the gene of interest. This 
        will be centered in the returned sequence.
        - ohe: bool, whether the input is one-hot encoded or not. Default = True.
        - rev_comp: bool, whether to get the reverse complement of the sequence. Note this 
        will not affect the strand of the gene as the returned sequence will be the reversed
        and compliment DNA returned for negative strand anyway.
        - pad_seq: bool, whether to pad the sequence with N's if the sequence is smaller than
        the model's receptive field (when TSS is at start or end of chrom). Default = True.
        
        Returns:
        - seq: str or torch.tensor, shape=(len(alphabet), seq_len), the DNA sequence
        """
        #sort issue with even model receptive fields and rev complementing to get - strand genes:
        if strand == "-" and self.model_receptive_field%2 == 0:
            #Issue is that if the model's receptive field is even, then the TSS will be in the middle + 1
            #This is fine if consistent but if the gene is on the reverse strand, then the TSS will be in the middle - 1
            #To sort this, we need to make the receptive field odd for neg stran genes
            gene_start = gene_start+1
        #work out the start and end positions
        start = max(0, gene_start - self.model_receptive_field//2)
        #if padding needed because start is at the beginning of the chromosome, get amount
        pad_N_strt = max((gene_start - self.model_receptive_field//2)*-1, 0)
        #max end is the chromosome length
        chrom_len = self.genome_dat.get_reference_length(chrom)
        end = min(gene_start + (self.model_receptive_field//2)+self.mod, chrom_len)
        pad_N_end = max((gene_start + (self.model_receptive_field//2)+self.mod - chrom_len), 0)
        #get the sequence (with padding if necessary)
        if pad_seq:
            seq = ("N"*pad_N_strt)+self.genome_dat.fetch(chrom, start, end).upper()+("N"*pad_N_end)
        else: #no padding
            seq = self.genome_dat.fetch(chrom, start, end).upper()
        #convert to tensor if needed    
        if ohe:
            from tangermeme.utils import one_hot_encode
            seq = one_hot_encode(seq, force=True).unsqueeze(0)
        #get strand specific gene sequence
        if strand == "-":
            seq = self._reverse_complement_dna(seq, ohe=ohe)
            
        if rev_comp:
            seq = self._reverse_complement_dna(seq, ohe=ohe)
        
        return seq
    
    def _reverse_complement_dna(self, seq, ohe=True, allow_N=True):
        """
        Get the reverse complement of an input DNA sequence 
        whether it is character bases or one-hot encoded.
        
        Parameters:
        - seq: str or torch tensor of one-hot encoded DNA sequence.
            Dimensions: (batch_size, alphabet_size, motif_size).
        - ohe: bool, whether the input is one-hot encoded or not
        - allow_N: bool, optional
            Whether to allow the return of the character 'N' in the sequence, i.e.
            if pwm at a position is all 0's return N. Default is False.
        Returns:
        - rev_comp: str or torch tensor of one-hot encoded DNA sequence
        """
        if not ohe:
            seq = seq.upper()
            bases_hash = {
                "A": "T",
                "T": "A",
                "C": "G",
                "G": "C",
                "N": "N"
            }
            #reverse order and get complement
            rev_comp = "".join([bases_hash[s] for s in reversed(seq)])
        else:
            import torch
            #If tensor is just (alphabet, motif_size) then add batch axis
            if len(seq.shape) == 2:
                seq = seq.unsqueeze(0)
            #reverse compliment of seq
            rev_comp = torch.flip(seq, dims=[1, 2])
        
        return rev_comp
