# Training an SLM

This experiment implements and modifies Andrej Karpathy's nanoGPT to train a significantly smaller language model. Despite its small size, this model can generate relatively coherent text but struggles with instruction-following and accurate information retrieval. Typically, models with over a billion parameters are needed for consistently usable responses.

Datasets were sourced from Hugging Face. The model architecture consists of 123.59 million parameters, 12 layers with 12 heads per layer, and an embedding dimension of 768 across a vocabulary of 50304 tokens. Initial training excluded biases and used a dropout rate of 0.0, which was adjusted to 1.0 during fine-tuning. The dataset mix included various sets from GPT-4, GPT-3.5, Databricks Dolly, WizardLM, and others[^1].

## Installation

### Prerequisites

- Download the datasets mentioned in the `create_first_training_data.py` script and place them in `Data/Datasets`.
- Install all required libraries via pip. Ensure you have access to `huggingface-cli` and `wandb`.

### Steps

1. Clone the repository.
    ```sh
    git clone <repository_url>
    ```

2. Install the necessary dependencies.
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To train and use the model, execute the following steps:

### Prepare your training data

```sh
python create_first_training_data.py
```

This script will create tokenized training and validation data from various datasets like OpenOrca, WizardLM, etc.

### Train the model

```sh
python train.py
```

### Fine-tune the model

```sh
python finetune.py
```

### Run and evaluate the model

```sh
python run.py
```

### Example

```plaintext
<user>Hello, how are you?<end>
<assistant>I am an AI model here to assist you!
```

### Note

Ensure you input the correct model import paths in `finetune.py` and `run.py`.

## Scripts

- **create_first_training_data.py**: Responsible for creating the initial training and validation data by combining multiple datasets and tokenizing them.
- **train.py**: Performs the training loop with evaluation, learning rate decay, and logging. Responsible for saving checkpoints and managing model state during training.
- **finetune.py**: Continues training a pre-trained model with additional datasets, improving its performance on specific tasks or datasets.
- **run.py**: Loads the trained model allowing for interactive querying to evaluate model performance.

## Contribute

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create your feature branch.
    ```sh
    git checkout -b feature/AmazingFeature
    ```
3. Commit your changes.
    ```sh
    git commit -m 'Add some AmazingFeature'
    ```
4. Push to the branch.
    ```sh
    git push origin feature/AmazingFeature
    ```
5. Open a pull request.

## License

This project is licensed under the MIT License.

For any further questions, please refer to the documentation or contact the repository maintainer.

## Model Architecture and Training

The architecture of the model comprises 12 layers with 12 attention heads per layer, resulting in approximately 123.59 million parameters. The embedding dimension is set to 768, spread across a vocabulary of 50304 tokens. Initial training parameters excluded biases, with a dropout rate of 0.0. During fine-tuning, dropout was set to 1.0 to improve generalization and model robustness.

### Dataset and Training Procedure

#### Initial Training

Initial training runs utilized a learning rate (LR) and weight decay phase, maintaining a batch size of 1024 tokens and a block size of 4096 tokens. We trained for a varying number of tokens and epochs, experimenting with data sources from GPT-4, GPT-3.5, Databricks Dolly, WizardLM, MPT 7B, Baize, OpenOrca, and Vicunia.

#### Fine-Tuning

After preliminary training, fine-tuning was performed, adjusting the dropout rate and other hyperparameters to refine model performance on specific tasks or datasets.

### Results and Performance

While the trained model can generate coherent text to some extent, its performance on following instructions and retrieving accurate information is limited. This aligns with known challenges in training smaller language models, which typically require over a billion parameters to consistently yield usable responses.

The overall aim was to test the limits of a relatively small language model and observe the trade-offs in capability versus the model's size. This can potentially help in understanding the scalability and the architecture modifications required for better performance in resource-constrained environments.

## Acknowledgements

Special thanks to Andrej Karpathy for nanoGPT, which served as the basis for this project, and to the various dataset providers on Hugging Face who made this research possible.

For any further inquiries or detailed descriptions of script functionalities, please refer to the provided documentation or reach out to the maintainers of this repository.

[^1]: Refer to the dataset sources for detailed information.
