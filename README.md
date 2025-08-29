# PRvL: Quantifying the Capabilities and Risks of Large Language Models for PII Redaction

## About
PRvL is an open-source framework for **PII redaction with Large Language Models (LLMs)**. It provides fine-tuned models and evaluation tools to help practitioners redact sensitive information from unstructured text while preserving context and meaning. Unlike rule-based or NER approaches that fail to generalize across formats, PRvL leverages the contextual understanding of LLMs to deliver accurate, efficient, and privacy-aware redaction.

Designed for **secure, self-managed environments**, PRvL enables data owners to perform redaction without exposing sensitive content to third-party services. The framework also supports multiple inference settings, making it adaptable for research, deployment, and compliance needs.

---

## Requisites
- [Hugging Face](https://huggingface.co/) account and personal access key  
- Python (>=3.8 recommended)  
- At least **4 GB of GPU RAM**  

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/LeonG19/PRvL.git
cd PRvL
pip install -r requirements.txt
```

---

## Usage

1. **Download requirements**  
   Install all required libraries from `requirements.txt`.

2. **Obtain a model checkpoint**  
   Download the model checkpoint of your choice and place it in a known directory.

3. **Get an example file**  
   Download an example file to test the model. Ensure that you point the LoRA checkpoint to the directory where you saved the model checkpoints.

4. **Run redaction**  
   Follow the instructions in the example template to start redacting PII on your own text!

Example:

```bash
python examples/run_redaction.py   --model_checkpoint ./model_checkpoints/deep-dis-ITPN2-pii-lora   --input_file ./examples/sample_text.txt   --output_file ./examples/output_redacted.txt
```

---

## Supported Entities
The models can detect and redact the following entity types:

```
[STREET] [GEOCOORD] [USERNAME] [GIVENNAME1] [SOCIALNUMBER]
[GIVENNAME2] [TEL] [CARDISSUER] [TITLE] [PASSPORT]
[PASS] [COUNTRY] [SEX] [BOD] [LASTNAME3]
[TIME] [LASTNAME2] [IDCARD] [EMAIL] [BUILDING]
[IP] [CITY] [POSTCODE] [SECADDRESS] [STATE]
[LASTNAME1] [DATE] [DRIVERLICENSE]
```

---

## Evaluation
The repository includes an evaluation notebook to measure model performance.  

- The evaluation compares **LLM outputs against ground truth** using metrics such as redaction accuracy, semantic preservation, and leakage rates.  
- To run your own evaluation, you will need to provide a **CSV file** containing:
  - Your model’s responses  
  - Ground-truth labels  

Example workflow:
```bash
jupyter notebook evaluation/evaluate_responses.ipynb
```

---

## Repository Structure
```
evaluation/           - Code and notebooks for evaluating responses
examples/             - Example scripts and test files
model_checkpoints/    - Directory to store downloaded model weights
requirements.txt      - Python dependencies
README.md             - Project overview
```

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation
If you use PRvL in your research, please cite:

```bibtex
@article{garza2025prvl,
  title   = {PRvL: Quantifying the Capabilities and Risks of Large Language Models for PII Redaction},
  author  = {Garza, Leon and Collaborators},
  year    = {2025},
  journal = {arXiv preprint arXiv:XXXX.XXXXX}
}
```

---

## Contact
For questions, issues, or collaborations:  
**Leon Garza** – [GitHub Profile](https://github.com/LeonG19)  
