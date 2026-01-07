# pdfQA: Diverse, Challenging, and Realistic Question Answering over PDFs

PDFs are the second-most used document type on the internet (after HTML). Yet, existing QA datasets commonly start from text sources or only address specific domains. In this paper, we present pdfQA, a multi-domain 2K human-annotated (real-pdfQA) and 2K synthetic dataset (syn-pdfQA) differentiating QA pairs in ten complexity dimensions (e.g., file type, source modality, source position, answer type). We apply and evaluate quality and difficulty filters on both datasets, obtaining valid and challenging QA pairs. We answer the questions with open-source LLMs, revealing existing challenges that correlate with our complexity dimensions. pdfQA presents a basis for end-to-end QA pipeline evaluation, testing diverse skill sets and local optimizations (e.g., in information retrieval or parsing).

## Implementation Details

I have rewritten the code to address the critique points while maintaining the standalone nature of the simulation.

### Key Improvements

1.  **Fixed Coupling and Magic Numbers**:
    The `RAGRetriever._simulate_tokenization` method now dynamically accesses `self.model.vocab_size`. This ensures that the generated token indices are always within the valid range of the embedding layer, preventing `IndexError` regardless of how the model is initialized.

2.  **Optimized Memory Usage**:
    In `LayoutAwareTransformer.forward`, the attention mask expansion was corrected. Instead of expanding to `(Batch, Seq, Hidden)` (which scales quadratically with model size), it is expanded to `(Batch, Seq, 1)`. PyTorch's broadcasting mechanism efficiently handles the element-wise multiplication with the embeddings `(Batch, Seq, Hidden)` without allocating unnecessary memory.

3.  **Implemented Multimodality (Layout Awareness)**:
    I introduced a `LayoutAwareTransformer`. It now accepts a `bboxes` tensor alongside `input_ids`. A `bbox_projection` linear layer projects the 4 coordinate values (normalized) into the hidden dimension. These spatial embeddings are added to the token embeddings, fulfilling the paper's requirement for processing layout information (PDF bounding boxes) alongside text.

4.  **Improved Filter Logic**:
    The `QualityFilter` no longer uses arbitrary hashing. It now implements a keyword overlap heuristic (intersection of Question/Answer tokens with Retrieved Context tokens). This serves as a transparent, logical proxy for an entailment check in this simulated environment, making the pipeline's behavior predictable and debuggable.

