"""
Dataset implementations for the deblend package.
"""
import ast
from pathlib import Path
from typing import (Callable,
                    Optional,
                    Dict,
                    Union,
                    Any,
                    )

import pandas as pd
from PIL import Image

from utils.file import load_json
from .registry import register_dataset_type


class BaseDataset():
    def __init__(self,
                 dataset_path: Union[str, Path],
                 img_size: int,
                 annotations_path: Optional[Dict[str, Any]] = None,
                 transform: Optional[Callable] = None,
                 **kwargs):
        self.dataset_path = Path(dataset_path)
        self.annotations_path = Path(annotations_path)
        self.img_size = (img_size, img_size)
        self.transform = transform


@register_dataset_type('dc_tool')
class DataCreationToolDataset(BaseDataset):
    def __init__(self,
                 dataset_path: Union[str, Path],
                 img_size: int,
                 annotations_path: Optional[Dict[str, Any]] = None,
                 transform: Optional[Callable] = None,
                 **kwargs):
        super().__init__(dataset_path=dataset_path,
                         img_size=img_size,
                         annotations_path=annotations_path,
                         transform=transform)
        if self.annotations_path.suffix != '.csv':
            raise ValueError(f'Expecting ".csv" annotation file, got "{self.annotations_path.suffix}"')
        self.metadata = pd.read_csv(self.annotations_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self,
                    index: int
                    ) -> Dict[str, Any]:
        curr_pair = self.metadata.iloc[index]

        image1_path = self.dataset_path / curr_pair['img1_name']
        image1_class = curr_pair['img1_name'].split('.')[0]
        image1 = Image.open(image1_path).convert('RGB').resize(self.img_size)

        image2_path = self.dataset_path / curr_pair['img2_name']
        image2_class = curr_pair['img2_name'].split('.')[0]
        image2 = Image.open(image2_path).convert('RGB').resize(self.img_size)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return {'image1':image1,
                'image1_path':str(image1_path),
                'image1_class':image1_class,
                'image1_prompt':curr_pair['img1_prompt'],
                'image2':image2,
                'image2_path':str(image2_path),
                'image2_class':image2_class,
                'image2_prompt':curr_pair['img2_prompt'],
                'index':index
                }

@register_dataset_type('BID')
class BIDDataset(BaseDataset):
    def __init__(self,
                 dataset_path: Union[str, Path],
                 img_size: int,
                 annotations_path: Optional[Dict[str, Any]] = None,
                 transform: Optional[Callable] = None,
                 **kwargs):
        super().__init__(dataset_path=dataset_path,
                         img_size=img_size,
                         annotations_path=annotations_path,
                         transform=transform)
        if self.annotations_path.suffix != '.csv':
            raise ValueError(f'Expecting ".csv" annotation file, got "{self.annotations_path.suffix}"')

        self.metadata = pd.read_csv(self.annotations_path)
        self.real_A_path = self.dataset_path / 'real_A'
        self.real_B_path = self.dataset_path / 'real_B'

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self,
                    index: int
                    ) -> Dict[str, Union[str, int]]:
        curr_row = self.metadata.iloc[index]
        curr_name = curr_row['image_name']
        image1_path = self.real_A_path / curr_name
        image1 = Image.open(image1_path).convert('RGB').resize(self.img_size)
        image2_path = self.real_B_path / curr_name
        image2 = Image.open(image2_path).convert('RGB').resize(self.img_size)
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return {'image1': image1,
                'image1_path': str(image1_path),
                'image1_class': 'Fruit',
                'image1_prompt': curr_row['prompt1'],
                'image2': image2,
                'image2_path': str(image2_path),
                'image2_class': 'Animal',
                'image2_prompt': curr_row['prompt2'],
                'index': index,
                'curr_name': curr_name,
                }



@register_dataset_type('dc_bid_tool')
class DCBidToolDataset(BaseDataset):
    def __init__(self,
                 dataset_path: Union[str, Path],
                 img_size: int,
                 annotations_path: Optional[Dict[str, Any]] = None,
                 transform: Optional[Callable] = None,
                 **kwargs):
        super().__init__(dataset_path=dataset_path,
                         img_size=img_size,
                         annotations_path=annotations_path,
                         transform=transform)
        if self.annotations_path.suffix != '.csv':
            raise ValueError(f'Expecting ".csv" annotation file, got "{self.annotations_path.suffix}"')
        self.metadata = pd.read_csv(self.annotations_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self,
                    index: int
                    ) -> Dict[str, Any]:
        curr_pair = self.metadata.iloc[index]

        image1_path = self.dataset_path / curr_pair['img1_path']
        image1_class = curr_pair['img1_class']
        image1 = Image.open(image1_path).convert('RGB').resize(self.img_size)

        image2_path = self.dataset_path / curr_pair['img2_path']
        image2_class = curr_pair['img2_class']
        image2 = Image.open(image2_path).convert('RGB').resize(self.img_size)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        coefficients = ['a', 'b', 'c', 'd', 'e', 'f']
        if all(name in curr_pair.index for name in coefficients):
            alpha_gt = curr_pair[coefficients].values.tolist()
        elif 'alpha' in curr_pair.index:
            alpha_gt = [float(curr_pair['alpha']), 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            raise ValueError('Annotation row must contain alpha or coefficients a-f')
        return {'image1':image1,
                'image1_path':str(image1_path),
                'image1_class':image1_class,
                'image1_prompt':curr_pair['img1_prompt'],
                'image2':image2,
                'image2_path':str(image2_path),
                'image2_class':image2_class,
                'image2_prompt':curr_pair['img2_prompt'],
                'index':index,
                'alpha_gt': alpha_gt,
                'pair_seeds': ast.literal_eval(curr_pair['z_seeds']),
                'pair_idx': index,
                }

@register_dataset_type('dc_bid_tool_second_stage')
class DCBidToolSecondStageDataset(BaseDataset):
    def __init__(self,
                 dataset_path: Union[str, Path],
                 img_size: int,
                 annotations_path: Optional[Dict[str, Any]] = None,
                 transform: Optional[Callable] = None,
                 **kwargs):
        super().__init__(dataset_path=dataset_path,
                         img_size=img_size,
                         annotations_path=annotations_path,
                         transform=transform)
        if self.annotations_path.suffix != '.csv':
            raise ValueError(f'Expecting ".csv" annotation file, got "{self.annotations_path.suffix}"')
        self.metadata = pd.read_csv(self.annotations_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self,
                    index: int
                    ) -> Dict[str, Any]:
        curr_pair = self.metadata.iloc[index]
        sample_path = Path(curr_pair['sample1_path']).parent
        alphas_dict = load_json(str(Path(sample_path) / 'alphas.json'))

        image1_path = curr_pair['image1_path']
        image1_class = curr_pair['image1_class']
        image1 = Image.open(image1_path).convert('RGB').resize(self.img_size)

        sample1_path = Path(sample_path) / 'sample1.png'
        sample1 = Image.open(sample1_path).convert('RGB').resize(self.img_size)

        image2_path = curr_pair['image2_path']
        image2_class = curr_pair['image2_class']
        image2 = Image.open(image2_path).convert('RGB').resize(self.img_size)

        sample2_path = Path(sample_path) / 'sample2.png'
        sample2 = Image.open(sample2_path).convert('RGB').resize(self.img_size)



        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        alpha_gt = alphas_dict['gt']['parameters']
        alpha_pred = alphas_dict['pred']['parameters']


        return {'image1': image1,
                'image1_path': str(image1_path),
                'image1_class': image1_class,
                'image1_prompt': curr_pair['image1_prompt'],
                'image2': image2,
                'image2_path': str(image2_path),
                'image2_class': image2_class,
                'image2_prompt': curr_pair['image2_prompt'],
                'sample1': sample1,
                'sample1_path': str(sample1_path),
                'sample2': sample2,
                'sample2_path': str(sample2_path),
                'index': index,
                'alpha_gt_type':  alphas_dict['gt']['type'],
                'alpha_gt': alpha_gt,
                'alpha_pred_type': alphas_dict['pred']['type'],
                'alpha_pred': alpha_pred,
                'smp_seed': curr_pair['smp_seed'],
                # Keep source pixels and alpha unchanged. Stage II uses this
                # selector decision only to align text embeddings to branches.
                'assignment_swapped': curr_pair.get('assignment_swapped', False),
                'pair_idx': int(curr_pair['pair'].split('_')[1]),
                'smp_idx': int(curr_pair['sample'].split('_')[1]),
                'details': load_json(str(Path(sample_path) / 'details.json'))
                }


@register_dataset_type('dc_bid_tool_second_stage_general_05')
class DCBidToolSecondStageGeneral05Dataset(BaseDataset):
    def __init__(self,
                 dataset_path: Union[str, Path],
                 img_size: int,
                 annotations_path: Optional[Dict[str, Any]] = None,
                 transform: Optional[Callable] = None,
                 **kwargs):
        super().__init__(dataset_path=dataset_path,
                         img_size=img_size,
                         annotations_path=annotations_path,
                         transform=transform)
        if self.annotations_path.suffix != '.csv':
            raise ValueError(f'Expecting ".csv" annotation file, got "{self.annotations_path.suffix}"')
        self.metadata = pd.read_csv(self.annotations_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self,
                    index: int
                    ) -> Dict[str, Any]:
        curr_pair = self.metadata.iloc[index]
        sample_path = Path(curr_pair['sample1_path']).parent
        alphas_dict = load_json(str(Path(sample_path) / 'alphas.json'))

        image1_path = curr_pair['image1_path']
        image1_class = curr_pair['image1_class']
        image1 = Image.open(image1_path).convert('RGB').resize(self.img_size)

        sample1_path = Path(sample_path) / 'sample1.png'
        sample1 = Image.open(sample1_path).convert('RGB').resize(self.img_size)

        image2_path = curr_pair['image2_path']
        image2_class = curr_pair['image2_class']
        image2 = Image.open(image2_path).convert('RGB').resize(self.img_size)

        sample2_path = Path(sample_path) / 'sample2.png'
        sample2 = Image.open(sample2_path).convert('RGB').resize(self.img_size)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        alpha_gt = [0.5, 0, 0, 0, 0, 0]
        alpha_pred = [0.5, 0, 0, 0, 0, 0]

        return {'image1': image1,
                'image1_path': str(image1_path),
                'image1_class': image1_class,
                'image1_prompt': curr_pair['image1_prompt'],
                'image2': image2,
                'image2_path': str(image2_path),
                'image2_class': image2_class,
                'image2_prompt': curr_pair['image2_prompt'],
                'sample1': sample1,
                'sample1_path': str(sample1_path),
                'sample2': sample2,
                'sample2_path': str(sample2_path),
                'index': index,
                'alpha_gt_type': alphas_dict['gt']['type'],
                'alpha_gt': alpha_gt,
                'alpha_pred_type': alphas_dict['pred']['type'],
                'alpha_pred': alpha_pred,
                'smp_seed': curr_pair['smp_seed'],
                'assignment_swapped': curr_pair.get('assignment_swapped', False),
                'pair_idx': int(curr_pair['pair'].split('_')[1]),
                'smp_idx': int(curr_pair['sample'].split('_')[1]),
                'details': load_json(str(Path(sample_path) / 'details.json'))
                }
