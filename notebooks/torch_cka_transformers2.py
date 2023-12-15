import torch 
import matplotlib.pyplot as plt
import re  
import argparse
import os 
import json 

from torch import nn 
from torch.utils.data import DataLoader, Subset 
from torch_cka.utils import add_colorbar
from tqdm import tqdm
from functools import partial
from warnings import warn
from typing import List, Dict
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset



class CKAGatherer: 
    """
    This class is an iterator that takes in two dataloaders and the corresponding models. at each iteration it yields 
    a tuple of return objects from the two models.
    """
    def __init__(
            self, 
            model1: nn.Module,
            model2: nn.Module,
            dataloader1: DataLoader,
            dataloader2: DataLoader = None,
            device: str = 'cpu', 
            model1_mode: bool = 'hf', 
            model2_mode: bool = 'hf',
            ):
        """
        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param dataloader1: (DataLoader) Dataloader for model 1
        :param dataloader2: (DataLoader) Dataloader for model 2. If not given, dataloader1 will be used for both models.
        :param device: Device to run the model
        """
        self.model1 = model1
        self.model2 = model2
        self.model1_mode = model1_mode
        self.model2_mode = model2_mode
        self.device = device

        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        self.model1.eval()
        self.model2.eval()

        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2 if dataloader2 is not None else dataloader1

        self.iterator1 = iter(self.dataloader1)
        self.iterator2 = iter(self.dataloader2)

    def __iter__(self):
        return self

    def next_hf(self, model, iterator): 
        input_dict = next(iterator)
        input_ids, attention_mask = input_dict['input_ids'], input_dict['attention_mask']
        return model.generate(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device), max_new_tokens=10)
    
    def next(self, model, iterator, model_mode): 
        if model_mode == 'hf': 
            return self.next_hf(model, iterator)
        else: 
            return model(next(iterator).to(self.device))

    def __next__(self):
        res1 = self.next(self.model1, self.iterator1, self.model1_mode)
        res2 = self.next(self.model2, self.iterator2, self.model2_mode)
        return res1, res2


class CKA:
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str ='cpu'):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2

        self.device = device

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"Both model have identical names - {self.model2_info['Name']}. " \
                 "It may cause confusion when interpreting the results. " \
                 "Consider giving unique names to the models :)")

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self.model1_features = {}
        self.model2_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_layers = model1_layers

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)")

        self.model2_layers = model2_layers

        self._insert_hooks()
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        self.model1.eval()
        self.model2.eval()

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model1":
            self.model1_features[name] = out

        elif model == "model2":
            self.model2_features[name] = out

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

        # Model 2
        for name, layer in self.model2.named_modules():
            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model2", name))
            else:

                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model2", name))

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def compare(
            self,
            dataloader1: DataLoader,
            dataloader2: DataLoader = None, 
        ) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.hsic_matrix = torch.zeros(N, M, 3)

        num_batches = min(len(dataloader1), len(dataloader1))

        gatherer = CKAGatherer(self.model1, self.model2, dataloader1, dataloader2, self.device)
        for _ in tqdm(gatherer, desc="| Comparing features |", total=num_batches):
            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                X = feat1.flatten(1)
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    Y = feat2.flatten(1)
                    L = Y @ Y.t()
                    L.fill_diagonal_(0.0)
                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())

        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "CKA": self.hsic_matrix,
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],
        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None, 
                     show: bool = False):
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if show is True:
            plt.show()


def get_next_version_dir(directory):
    if not os.path.exists(directory):
        # Directory doesn't exist, return with version_0
        return f"{directory}/version_0"

    # List all items in the directory
    items = os.listdir(directory)

    # Filter out items that are not directories or don't follow the version_i pattern
    version_dirs = [item for item in items if os.path.isdir(os.path.join(directory, item)) and item.startswith("version_")]

    if not version_dirs:
        # No version directories, return with version_0
        return f"{directory}/version_0"

    # Extract version numbers and find the highest one
    version_numbers = [int(dir.split("_")[-1]) for dir in version_dirs]
    max_version = max(version_numbers)

    # Return directory with version_{max_version+1}
    return f"{directory}/version_{max_version+1}"


def load_translation_dataset():
    # We test representations on a translation task.
    dataset = load_dataset('glue', 'mrpc', split='test')
    def prepend_translation_task(example):
        example["sentence1"] = "Translate from English to French: " + example["sentence1"]
        return example
    dataset = dataset.map(prepend_translation_task)
    return dataset


def load_model_and_tokenizer(name='t5-small'):
    tokenizer = T5Tokenizer.from_pretrained(name)
    model = T5ForConditionalGeneration.from_pretrained(name)
    return model, tokenizer 


def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch['sentence1'], padding='max_length', truncation=True)
    dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset 


def get_dataloader(dataset, batch_size: int = 8, num_batches: int = -1, drop_last=True):
    # Maybe take a subset e.g. for rapid testing 
    if num_batches >= 0: 
        dataset = Subset(dataset, indices=range(0, num_batches * batch_size))
    return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last) 


def match_module_names_from_pattern(model: nn.Module, pattern: str):
    if isinstance(pattern, str): 
        pattern = re.compile(pattern)
    return [name for name, _ in model.named_modules() if pattern.match(name)]


def get_pattern(encoder_self_attention=True, 
                encoder_fc=True, 
                decoder_self_attention=True, 
                decoder_cross_attention=True, 
                decoder_fc=True):
    patterns = []

    # Patterns for encoder
    if encoder_self_attention:
        patterns.append(r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.(q|k|v|o)')
    if encoder_fc:
        # wi_0, wi_1 is for flan_t5 
        patterns.append(r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.(wi|wi_0|wi_1|wo)')

    # Patterns for decoder
    if decoder_self_attention:
        patterns.append(r'decoder\.block\.[1-9]\d*\.layer\.\d+\.SelfAttention\.(q|k|v|o)')
    if decoder_cross_attention:
        patterns.append(r'decoder\.block\.[1-9]\d*\.layer\.\d+\.EncDecAttention\.(q|k|v|o)')
    if decoder_fc:
        patterns.append(r'decoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.(wi|wo)')

    return '|'.join(patterns)


def get_module_names(
    module,
    encoder_self_attention=True, 
    encoder_fc=True, 
    decoder_self_attention=True, 
    decoder_cross_attention=True, 
    decoder_fc=True, 
):
    return match_module_names_from_pattern(
        module, 
        get_pattern(encoder_self_attention, encoder_fc, decoder_self_attention, decoder_cross_attention, decoder_fc))


def main(
        model1_name: str, 
        model2_name: str, 
        batch_size: int = 8, 
        num_batches: int = -1, 
        device: str = None, 
        save_path: str = './',
    ): 
    if device is None: 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"DEVICE SET TO {device.upper()}".center(100, "-"))

    dataset = load_translation_dataset()
 
    model1, tokenizer1 = load_model_and_tokenizer(model1_name)
    model2, tokenizer2 = load_model_and_tokenizer(model2_name)
    model1_layers = get_module_names(model1)
    model2_layers = get_module_names(model2)
    
    cka = CKA(
        model1, 
        model2, 
        model1_name=model1_name, 
        model2_name=model2_name, 
        model1_layers=model1_layers, 
        model2_layers=model2_layers,
        device=device, 
    )

    dataloader1 = get_dataloader(
        dataset=tokenize_dataset(dataset, tokenizer1), 
        batch_size=batch_size, 
        num_batches=num_batches,
    )
    dataloader2 = get_dataloader(
        dataset=tokenize_dataset(dataset, tokenizer2), 
        batch_size=batch_size, 
        num_batches=num_batches,
    )

    print(f"{len(dataloader1)=}, {len(dataloader2)=}")

    cka.compare(dataloader1=dataloader1, dataloader2=dataloader2)

    # log and plot 
    title = f"{model1_name}-vs-{model2_name.replace('/', '_')}"
    directory = os.path.join(save_path, title)
    save_path = get_next_version_dir(directory)
    os.makedirs(save_path, exist_ok=False) # This directory should not exist, since we just created the next version

    # Save plot and cka results
    cka.plot_results(save_path=os.path.join(save_path, 'heatmap.png'))
    torch.save(cka.export(), os.path.join(save_path, 'cka.pt'))
    # with open(os.path.join(save_path, 'cka.json'), 'w') as f: 
    #     json.dump(cka.export(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two models using CKA.")
    parser.add_argument("--model1", default='t5-small', help="Path to the first model", type=str)
    parser.add_argument("--model2", default='t5-small', help="Path to the second model", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_batches", default=-1, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--save_path", default='./', type=str)

    args = parser.parse_args()
    model1_name = args.model1
    model2_name = args.model2 
    batch_size = args.batch_size
    num_batches = args.num_batches 
    device = args.device 
    print(f"ARGUMENTS: {model1_name=}, {model2_name=}, {batch_size=}, {num_batches=}, {device=}")

    main(args.model1, args.model2, args.batch_size, args.num_batches, args.device)