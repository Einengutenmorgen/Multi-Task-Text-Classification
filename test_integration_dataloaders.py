#test_integration_dataloaders_real.py

import unittest
import os
import torch
from torch.utils.data import DataLoader
import collections

# --- Import the script to be tested ---
try:
    import data_loading
except ImportError:
    print("Error: Could not import 'data_loading.py'.")
    print("Please save the provided script as 'data_loading.py' in the same directory.")
    exit(1)


class TestDataLoaderWithRealData(unittest.TestCase):
    """
    Integration tests using REAL data files.
    
    REQUIREMENTS:
    - All datasets must be downloaded and placed in their expected locations
    - Paths in data_loading.py must be correct for your system
    - This will take longer to run than mock tests
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize once before all tests.
        Sets up the dataset and checks that data files exist.
        """
        cls.tokenizer_name = 'distilbert-base-uncased'
        cls.max_length = 128
        
        print("\n" + "="*70)
        print("INITIALIZING DATASET WITH REAL DATA")
        print("="*70)
        
        # Initialize the dataset - this will print loading status
        cls.dataset = data_loading.UnifiedDataset(
            tokenizer_name=cls.tokenizer_name, 
            max_length=cls.max_length
        )
        
        print("="*70)
        print(f"TOTAL SAMPLES LOADED: {len(cls.dataset)}")
        print("="*70 + "\n")
        
        # Store counts for validation
        cls.task_counts = {
            task: len(data['texts']) 
            for task, data in cls.dataset.task_data.items()
        }
        
        # Verify at least some data loaded
        total_samples = sum(cls.task_counts.values())
        if total_samples == 0:
            raise RuntimeError(
                "No data loaded! Please check that all dataset files exist "
                "and paths in data_loading.py are correct."
            )
        
        # Warn about missing datasets
        missing_datasets = [
            task for task, count in cls.task_counts.items() 
            if count == 0
        ]
        if missing_datasets:
            print(f"WARNING: The following datasets have 0 samples: {missing_datasets}")
            print("Some tests may be skipped.\n")
    
    # --- Helper Methods ---
    
    def _find_task_sample(self, task_name):
        """Helper to find the first sample index for a given task."""
        for idx, (task, _) in enumerate(self.dataset.task_indices):
            if task == task_name:
                return idx
        return None
    
    def _skip_if_task_empty(self, task_name):
        """Skip test if the specified task has no data."""
        if self.task_counts.get(task_name, 0) == 0:
            self.skipTest(f"{task_name} dataset not loaded or empty")
    
    # --- Test Cases ---
    
    def test_01_dataset_initialization_and_counts(self):
        """Tests if the dataset initializes and reports counts."""
        # Check that dataset length equals sum of task counts
        total = sum(self.task_counts.values())
        self.assertEqual(len(self.dataset), total)
        
        # Check that task_indices has correct length
        self.assertEqual(len(self.dataset.task_indices), total)
        
        # Check that all task names in task_data are in SCHEMA
        for task_name in self.dataset.task_data.keys():
            self.assertIn(task_name, data_loading.SCHEMA)
        
        # Print summary
        print("\n--- Dataset Summary ---")
        for task, count in self.task_counts.items():
            print(f"{task:15s}: {count:,} samples")
        print(f"{'TOTAL':15s}: {total:,} samples\n")

    def test_02_getitem_jigsaw(self):
        """Tests a Jigsaw sample structure."""
        self._skip_if_task_empty('jigsaw')
        
        idx = self._find_task_sample('jigsaw')
        self.assertIsNotNone(idx, "Could not find Jigsaw sample")
        
        sample = self.dataset[idx]
        
        # Check that Jigsaw labels are valid floats (0 or 1)
        self.assertTrue(torch.is_tensor(sample['labels_jigsaw']))
        self.assertEqual(sample['labels_jigsaw'].dtype, torch.float)
        self.assertEqual(
            len(sample['labels_jigsaw']), 
            len(data_loading.SCHEMA['jigsaw'])
        )
        
        # Check all values are either 0, 1, or -100
        unique_values = sample['labels_jigsaw'].unique()
        for val in unique_values:
            self.assertIn(val.item(), [-100.0, 0.0, 1.0])
        
        # Check that at least ONE label is not -100 (it's a Jigsaw sample)
        self.assertTrue(torch.any(sample['labels_jigsaw'] != -100))
        
        # Check other task labels are all -100
        self.assertTrue(torch.all(sample['labels_goemotions'] == -100))
        self.assertEqual(sample['labels_davidson'].item(), -100)
        self.assertEqual(sample['labels_olid'].item(), -100)
        self.assertEqual(sample['labels_rumour'].item(), -100)

    def test_03_getitem_goemotions(self):
        """Tests a GoEmotions sample structure."""
        self._skip_if_task_empty('goemotions')
        
        idx = self._find_task_sample('goemotions')
        self.assertIsNotNone(idx)
        
        sample = self.dataset[idx]
        
        # Check GoEmotions labels
        self.assertTrue(torch.is_tensor(sample['labels_goemotions']))
        self.assertEqual(sample['labels_goemotions'].dtype, torch.float)
        self.assertEqual(
            len(sample['labels_goemotions']), 
            len(data_loading.SCHEMA['goemotions'])
        )
        
        # Check at least one label is not -100
        self.assertTrue(torch.any(sample['labels_goemotions'] != -100))
        
        # Check other labels are -100
        self.assertTrue(torch.all(sample['labels_jigsaw'] == -100))
        self.assertEqual(sample['labels_davidson'].item(), -100)

    def test_04_getitem_davidson(self):
        """Tests a Davidson sample structure."""
        self._skip_if_task_empty('davidson')
        
        idx = self._find_task_sample('davidson')
        self.assertIsNotNone(idx)
        
        sample = self.dataset[idx]
        
        # Check Davidson label is a valid class (0, 1, or 2)
        self.assertTrue(torch.is_tensor(sample['labels_davidson']))
        self.assertEqual(sample['labels_davidson'].dtype, torch.long)
        self.assertIn(sample['labels_davidson'].item(), [0, 1, 2])
        
        # Check other labels
        self.assertTrue(torch.all(sample['labels_jigsaw'] == -100))
        self.assertTrue(torch.all(sample['labels_goemotions'] == -100))

    def test_05_getitem_olid(self):
        """Tests an OLID sample structure."""
        self._skip_if_task_empty('olid')
        
        idx = self._find_task_sample('olid')
        self.assertIsNotNone(idx)
        
        sample = self.dataset[idx]
        
        # Check OLID label is valid (0 or 1)
        self.assertEqual(sample['labels_olid'].dtype, torch.long)
        self.assertIn(sample['labels_olid'].item(), [0, 1])
        
        # Check other labels
        self.assertEqual(sample['labels_davidson'].item(), -100)
        self.assertEqual(sample['labels_rumour'].item(), -100)

    def test_06_getitem_rumour(self):
        """Tests a RumourEval sample (text_a + text_b)."""
        self._skip_if_task_empty('rumour')
        
        idx = self._find_task_sample('rumour')
        self.assertIsNotNone(idx)
        
        sample = self.dataset[idx]
        
        # Check Rumour label is valid (0, 1, 2, or 3)
        self.assertIn(sample['labels_rumour'].item(), [0, 1, 2, 3])
        
        # Check other labels
        self.assertEqual(sample['labels_davidson'].item(), -100)
        self.assertTrue(torch.all(sample['labels_jigsaw'] == -100))
        
        # Check tokenization includes [SEP] token (for text_a + text_b)
        decoded = self.dataset.tokenizer.decode(
            sample['input_ids'], 
            skip_special_tokens=False
        )
        self.assertIn("[SEP]", decoded)

    def test_07_task_sampler_epoch_length(self):
        """Tests the TaskSampler calculates correct epoch length."""
        sampler = data_loading.TaskSampler(self.dataset)
        
        # Find max task size
        max_size = max(
            len(self.dataset.task_data[task]['texts'])
            for task in sampler.task_names
        )
        
        num_tasks = len(sampler.task_names)
        expected_epoch_size = max_size * num_tasks
        
        self.assertEqual(sampler.max_task_size, max_size)
        self.assertEqual(len(sampler), expected_epoch_size)
        
        print(f"\n--- TaskSampler Info ---")
        print(f"Tasks: {sampler.task_names}")
        print(f"Max task size: {max_size:,}")
        print(f"Epoch size: {expected_epoch_size:,}\n")

    def test_08_dataloader_batch_shapes(self):
        """Tests the DataLoader produces correct batch shapes."""
        sampler = data_loading.TaskSampler(self.dataset)
        batch_size = 16
        
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            sampler=sampler, 
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        
        # Check batch size
        self.assertEqual(batch['input_ids'].shape[0], batch_size)
        
        # Check tensor shapes
        self.assertEqual(
            batch['input_ids'].shape, 
            (batch_size, self.max_length)
        )
        self.assertEqual(
            batch['attention_mask'].shape, 
            (batch_size, self.max_length)
        )
        
        # Label shapes
        num_jigsaw = len(data_loading.SCHEMA['jigsaw'])
        num_goemotions = len(data_loading.SCHEMA['goemotions'])
        
        self.assertEqual(
            batch['labels_jigsaw'].shape, 
            (batch_size, num_jigsaw)
        )
        self.assertEqual(
            batch['labels_goemotions'].shape, 
            (batch_size, num_goemotions)
        )
        self.assertEqual(batch['labels_davidson'].shape, (batch_size,))
        self.assertEqual(batch['labels_olid'].shape, (batch_size,))
        self.assertEqual(batch['labels_rumour'].shape, (batch_size,))
        
        # Check dtypes
        self.assertEqual(batch['labels_jigsaw'].dtype, torch.float)
        self.assertEqual(batch['labels_goemotions'].dtype, torch.float)
        self.assertEqual(batch['labels_davidson'].dtype, torch.long)
        self.assertEqual(batch['labels_olid'].dtype, torch.long)
        self.assertEqual(batch['labels_rumour'].dtype, torch.long)

    def test_09_task_sampler_distribution(self):
        """
        Tests that TaskSampler balances tasks by sampling each
        task max_task_size times per epoch.
        """
        sampler = data_loading.TaskSampler(self.dataset)
        
        # Get max task size
        max_size = sampler.max_task_size
        
        # Create lookup: global index -> task name
        task_lookup = {
            i: task_name 
            for i, (task_name, _) in enumerate(self.dataset.task_indices)
        }
        
        # Get all indices from one epoch
        indices = list(iter(sampler))
        
        # Count tasks
        task_counts = collections.Counter(
            task_lookup[i] for i in indices
        )
        
        print("\n--- Task Distribution in Epoch ---")
        for task in sorted(task_counts.keys()):
            print(f"{task:15s}: {task_counts[task]:,} samples")
        print(f"{'Expected each':15s}: {max_size:,} samples\n")
        
        # Check that each task appears exactly max_size times
        for task_name in sampler.task_names:
            self.assertEqual(
                task_counts[task_name], 
                max_size,
                f"Task {task_name} not balanced correctly"
            )
        
        # Check total
        self.assertEqual(len(indices), max_size * len(sampler.task_names))

    def test_10_input_ids_tokenization(self):
        """Tests that input_ids are correctly tokenized."""
        # Get a sample from each task (if available)
        for task_name in ['jigsaw', 'goemotions', 'davidson', 'olid', 'rumour']:
            if self.task_counts.get(task_name, 0) == 0:
                continue
                
            idx = self._find_task_sample(task_name)
            sample = self.dataset[idx]
            
            # Check that input_ids is correct shape
            self.assertEqual(
                sample['input_ids'].shape[0], 
                self.max_length
            )
            
            # Check that it starts with [CLS] token
            cls_token_id = self.dataset.tokenizer.cls_token_id
            self.assertEqual(sample['input_ids'][0].item(), cls_token_id)
            
            # Check that [SEP] appears
            sep_token_id = self.dataset.tokenizer.sep_token_id
            self.assertIn(
                sep_token_id, 
                sample['input_ids'].tolist(),
                f"[SEP] token not found in {task_name} sample"
            )
            
            # Check attention mask matches input_ids
            # (1 for real tokens, 0 for padding)
            pad_token_id = self.dataset.tokenizer.pad_token_id
            for i in range(self.max_length):
                token_id = sample['input_ids'][i].item()
                attention = sample['attention_mask'][i].item()
                
                if token_id == pad_token_id:
                    self.assertEqual(attention, 0)
                else:
                    self.assertEqual(attention, 1)

    def test_11_label_consistency(self):
        """
        Tests that labels are consistent with the task schema
        and that -100 is used correctly for ignore_index.
        """
        # Check 10 random samples
        num_samples = min(10, len(self.dataset))
        indices = torch.randperm(len(self.dataset))[:num_samples].tolist()
        
        for idx in indices:
            sample = self.dataset[idx]
            task_name, _ = self.dataset.task_indices[idx]
            
            # For the current task, labels should NOT be all -100
            if task_name == 'jigsaw':
                self.assertFalse(
                    torch.all(sample['labels_jigsaw'] == -100),
                    "Jigsaw sample has all -100 labels"
                )
                # Other tasks should be -100
                self.assertTrue(torch.all(sample['labels_goemotions'] == -100))
                self.assertEqual(sample['labels_davidson'].item(), -100)
                
            elif task_name == 'goemotions':
                self.assertFalse(
                    torch.all(sample['labels_goemotions'] == -100)
                )
                self.assertTrue(torch.all(sample['labels_jigsaw'] == -100))
                
            elif task_name == 'davidson':
                self.assertNotEqual(sample['labels_davidson'].item(), -100)
                self.assertTrue(torch.all(sample['labels_jigsaw'] == -100))
                
            elif task_name == 'olid':
                self.assertNotEqual(sample['labels_olid'].item(), -100)
                self.assertEqual(sample['labels_davidson'].item(), -100)
                
            elif task_name == 'rumour':
                self.assertNotEqual(sample['labels_rumour'].item(), -100)
                self.assertEqual(sample['labels_davidson'].item(), -100)


class TestDatasetSchema(unittest.TestCase):
    """Additional tests for schema validation."""
    
    @classmethod
    def setUpClass(cls):
        cls.tokenizer_name = 'distilbert-base-uncased'
        cls.dataset = data_loading.UnifiedDataset(
            tokenizer_name=cls.tokenizer_name, 
            max_length=128
        )
    
    def test_schema_matches_labels(self):
        """Tests that SCHEMA matches actual label dimensions."""
        # For multi-label tasks
        for task in ['jigsaw', 'goemotions']:
            if len(self.dataset.task_data[task]['texts']) > 0:
                first_sample_labels = self.dataset.task_data[task]['labels'][0]
                schema_size = len(data_loading.SCHEMA[task])
                
                self.assertEqual(
                    len(first_sample_labels),
                    schema_size,
                    f"{task} label size mismatch"
                )
    
    def test_all_tasks_in_schema(self):
        """Tests that all tasks are defined in SCHEMA."""
        for task_name in self.dataset.task_data.keys():
            self.assertIn(
                task_name, 
                data_loading.SCHEMA,
                f"Task {task_name} not in SCHEMA"
            )


if __name__ == '__main__':
    # Print system info
    print("\n" + "="*70)
    print("REAL DATA INTEGRATION TESTS")
    print("="*70)
    print("\nREQUIREMENTS:")
    print("- All datasets must be downloaded")
    print("- Paths in data_loading.py must be correct")
    print("- This will take several minutes to run")
    print("="*70 + "\n")
    
    # Run tests with verbose output
    unittest.main(verbosity=2)