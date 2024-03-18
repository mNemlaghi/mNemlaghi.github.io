# LoRA fine-tuning a pre-trained embedding with SageMaker, Bedrock and Hugging Face

In [introduction](https://mnemlaghi.github.io/cloud-embeddings/), we gathered some historical perspective and highlighted the importance of word embeddings in our current era. [In part one](https://mnemlaghi.github.io/cloud-embeddings/part-one-evaluate) we selected and evaluated some embeddings.

This part is more destined to readers who are already familiarized with modern language model fine-tuning; therefore, if you want to deploy your application with state-of-the-art embedding without fine-tuning it, you can directly skip it and go directly to the next part.
Now we'll focus on how we can fine-tune it in a modern and cost effective manner with a single GPU on SageMaker, with LoRA (low-rank adapters). First, let's get practical by fetching a real-world dataset and a real-world measurement. Fasten your seatbelt, let's go!


## The business challenge

### Shopping queries dataset

[Amazon Shopping Queries Dataset](https://github.com/amazon-science/esci-data) is based on every online business utmost priority: improving customer experience. We've all been in a stage where, when searching for a specific product, suggested items were at the very least poorly relevant. It seemed that the website would either suggest irrelevant items or nothing, whereas the match was indeed in store… Shopping queries dataset is derived from real-world customer queries that can be challenging.
Indeed, Wayfair estimates that, on a monthly basis, 80% of user queries are previously unseen, and new products emerge everyday.
From the repository, Shopping Dataset contains <query, product> pairs, with a corresponding jugement, assessing matching level of the query:
- E stands for Exact
- S is a Substitute
- C is a Complement
- I is irrelevant

It also contains other fields that are beyond the scope of this blog.

### The metric: cosine similarity

Numerous resources can be found  on cosine similarity (look at [Google scholar results !](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=cosine+similarity+embeddings&btnG=)). Nevertheless, let's explain it briefly. It refers to the measure of similarity between two vectors in a high-dimensional space. Remember, in word embeddings, words are represented as vectors in a high-dimensional space such that similar words are mapped to nearby points in that space. 

The cosine similarity between two word vectors is calculated as the dot product of the two vectors divided by the product of their magnitudes. This produces a score ranging from -1 (completely dissimilar) to 1 (completely similar). In our case, we would like to optimize the relevance between a product embedding, _p_ and a query embedding, _q_.  The formula for calculating cosine similarity is _cos(θ) = (p ⋅ q) / (|q| |p|)_  where θ is the angle between them.

We also need to check whether computed similarity for a batch of examples is whether correct or incorrect. Let's continue with the following assumption: _Exact_ and _Substitute_ will be marked as relevant `1` and the rest (_complement_ and _irrelevant_) will be marked as irrelevant `-1` . 

The loss will then be computed as follows:

*  _1 - cos(p,q)_ if label is _E_ or _S_
*  _max(cos(p,q), epsilon)_ otherwise, where _epsilon_ is a very small number. 

Last word on cosine similarity:  this [Netflix paper](https://arxiv.org/abs/2403.05440), dating from March 2024, arguably claims that blindly relying on cosine similarity could have side effects, and advice data standardization.

## The method
### Parameter-efficient fine-tuning

PEFT stands for parameter efficient fine-tuning. Classically, when we fine-tune a model, we change every model parameter. A naive vision of this approach could lead to the following caveats:

- Prohibitive costs. In a scarce resource environment, when model is very large

- Catastrophic forgetting: this phenomenon happens when fine-tuning changes parameters to the point the fine-tuned model forgets previously encapsulated general information

PEFT circumvents these issues by freezing a majority of layers in pre-trained neural networks and focus only on specific ones. PEFT acts as an umbrella of techniques; LoRA is one of the best-in-class.

### Primer on Low Rank Adapters

In the context of Machine Learning, LoRA (Low Rank Adapters) is a flavor of parameter efficient fine tuning. It is a cost-efficient way to fine-tune a model. In a nutshell, during fine-tuning, Lora constraints pre-trained weights with matrices called update matrices with a rank called r, thanks to low-rank approximations. These lower rank matrices, with lower rank, are then put inside the network during fine-tuning. After performing finetuning, initial matrices are reconstructed with updates provided by update matrices. (source: https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)

### What happens behind the scenes?
A neural network accepts an input x and outputs an output h. Pre-trained weights are represented by a matrix W₀ ∈ ℝᵈˣᵈ.
We add a new set of parameters, called adapters, A and B, whose dimensions are respectively d x rand r x d, with r << d.


During fine-tuning:
* pre-trained weights W₀ are frozen: it means their parameters don't change during backpropagation.
* Adapters A and B are trained; their parameters are changed.
* After fine-tuning, we reconstruct the new weights by merging pre-trained weights and adapter weights : W= W₀ + AB

Let's judge it by real numbers: if d=100, then pre-trained weights have 100000 parameters to train. If we choose r=10, then, A would have 10*100 and B would have 100*10 parameters to train, giving overall 2000 parameters, We'd use 20% of the GPU memory during backpropagation. This means that batch size can be higher when fine-tuning. Which means that training can happen faster.

### Why does this work?
Since pre-trained weights are frozen, they're not subject to back-propagation. Therefore, by design, LoRA consumes less amount of GPU memory
It focuses on previously chosen, specific layers; catastrophic forgetting is less likely to happen.

This method opens new perspectives in Machine Learning at the age of LLMs, as now, updates are more cost-efficient.

### Great! How can I implement this?

Short answer: you'd only need to implement HuggingFace for the implementation part (transformers and peft library), and AWS Sagemaker for spinning up and down the right resources. Here are the high level steps:

Write a HuggingFace fine-tuning script: you can find examples here or here; this script will probably contain a model object

```{python}
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(pretrained_model)
tokenizer = AutoModel.from_pretrained(pretrained_model)

# [...] YOUR Training script
We'll add a simple LoraConfig (from peft library) object. Choose carefully the rank and target modules, as they'll impact fine-tuning speed. Since we extract embeddings, task type is FEATURE_EXTRACTION

from peft import LoraConfig, get_peft_model, TaskType
lora_rank=8
config = LoraConfig(
                r=lora_rank,
                lora_alpha=32,
                target_modules=["key","query", "value"],
                bias="none",
                lora_dropout=0.05,
                inference_mode=False,
                task_type=TaskType.FEATURE_EXTRACTION
            )
Create peft model thanks to the model object and above configuration

model = get_peft_model(model, config)
```

Now you can fine-tune.

```{python}
## Merge the models with trained Adapters, and save it to /opt/ml/code
    merged = model.merge_and_unload()
    merged.save_pretrained(args.model_dir)
````


### Wrapping it up in SageMaker code

SageMaker accepts this script with slight variation. Create a requirements.txt to install required packages, mainly peft.

```
peft
accelerate==0.21.0
````

Then, we'll create a training script called train.py (example in repository). Nothing changes from your regular HuggingFace trainer, except with the right I/O configuration for SageMaker.
Then put both of these files into a directory.
```
./peftscripts
├── requirements.txt
└── train.py
````

We finish by invoking a training job with SageMaker SDK, with a set of naive hyper parameters. 

An important hyper parameter is `lora_rank`. It's a degree of reduction. A tradeoff is to be found between  compression and performance efficiency.

```{python}
from sagemaker.huggingface import HuggingFace
import sagemaker
import time
# hyperparameters, which are passed into the training job
hyperparameters={'pretrained_model': "TaylorAI/gte-tiny",
                 'batch_size': 120,
                 'eval_batch_size': 160,
                 'lora_rank':8, 
                 'epochs':1, 
                 'lr':1e-6
                 }
huggingface_estimator = HuggingFace(entry_point='train.py',
                            source_dir='./peftscripts',
                            instance_type='ml.g4dn.xlarge',
                            instance_count=1,
                            role = sagemaker.get_execution_role(),
                            transformers_version='4.26',
                            pytorch_version='1.13',
                            py_version='py39',
                            hyperparameters = hyperparameters, 
                             metric_definitions=[
                                 {'Name': 'training_loss', 'Regex': 'training loss ([0-9\\.]+)'},
                                 {'Name': 'eval_loss', 'Regex': 'eval loss ([0-9\\.]+)'}
                             ]
)
timing = str(int(time.time()))
huggingface_estimator.fit(job_name=f"PeftFTBGE{timing}", wait=False)
```

At the end of the training job, you'll get model archive in the default s3 bucket. _Et voilà_ !
Model artifacts will be uploaded into the following s3 URI.
s3://{YOUR_DEFAULT_BUCKET}/{YOUR_TRAINING_JOB_NAME}/output/model.tar.gz


## Troubleshooting and remarks
💡 Please note that `wait=False` parameter is intended to unblock the training job;
💡 The higher`lora_rank` is, the bigger will be the adapter matrices, the higher the training time.

## Recap and what's next

In this post, we chose a pre-trained embedding model amongst the best ones, and we delved into modern fine-tuning with parameter-efficient tuning, LoRA and SageMaker training.
Next, we'll see how we'll deploy in a cost effective - and repeatable - manner an embedding system with SageMaker Endpoint and CDK!

