import unittest

import numpy as np
import torch
from util.chat_input import ChatInput
from util.subject import batch_interventions, llama31_8B_instruct_config, make_subject


class TestSubject(unittest.TestCase):
    def setUp(self):
        self.subject = make_subject(llama31_8B_instruct_config)
        self.subject.model.generation_config.pad_token_id = self.subject.tokenizer.pad_token_id

    def test_batch_interventions(self):
        num_pad_tokens = [1, 0]

        interventions = [
            {(0, 0, 0): 1.0, (2, 3, 10): 4.4, (2, 4, 11): 5.5, (2, 10, 100): 100.0},
            {(0, 0, 0): 1.0, (1, 1, 0): 2.0, (2, 0, 13): 6.6, (2, 2, 15): 7.7},
        ]

        interventions_by_layer = batch_interventions(interventions, num_pad_tokens)

        correct_intervention_tensors_by_layer = {
            0: (
                torch.tensor([[0], [1]]),  # batch_idxs
                torch.tensor([[1], [0]]),  # tokens
                torch.tensor([[0], [0]]),  # neurons
                torch.tensor([[1.0], [1.0]]),  # values
            ),
            1: (
                torch.tensor([[1]]),  # batch_idxs
                torch.tensor([[1]]),  # tokens
                torch.tensor([[0]]),  # neurons
                torch.tensor([[2.0]]),  # values
            ),
            2: (
                torch.tensor([[0, 0, 0], [1, 1, 1]]),  # batch_idxs
                torch.tensor([[4, 5, 11], [0, 2, 2]]),  # tokens
                torch.tensor([[10, 11, 100], [13, 15, 15]]),  # neurons
                torch.tensor([[4.4, 5.5, 100.0], [6.6, 7.7, 7.7]]),  # values
            ),
        }
        for layer in interventions_by_layer:
            for i in range(len(interventions_by_layer[layer])):
                self.assertTrue(
                    torch.equal(
                        interventions_by_layer[layer][i],
                        correct_intervention_tensors_by_layer[layer][i],
                    )
                )

    def test_generate_batched(self):
        cis = [
            ChatInput(
                system_prompt=None,
                conversation=[{"role": "user", "content": "What is the capital of France?"}],
                use_chat_format=True,
            ),
            ChatInput(
                system_prompt=None,
                conversation=[{"role": "user", "content": "Hello what is the capital of France?"}],
                use_chat_format=True,
            ),
        ]

        interventions = [
            {(0, 0, 0): 1.0, (2, 3, 10): 4.4, (2, 4, 11): 5.5, (2, 10, 100): 100.0},
            {(0, 0, 0): 1.0, (1, 1, 0): 2.0, (2, 0, 13): 6.6, (2, 2, 15): 7.7},
        ]

        non_batched = [
            self.subject.generate(
                ci,
                max_new_tokens=3,
                num_return_sequences=1,
                neuron_interventions=intervs,
            )
            for ci, intervs in zip(cis, interventions)
        ]

        batched = self.subject.do_batched_interventions(
            cis,
            max_new_tokens=3,
            neuron_interventions=interventions,
        )

        for i in range(len(cis)):
            self.assertTrue(
                np.isclose(non_batched[i].output_ids_BT[0], batched[i].output_ids_BT).all()
            )
            self.assertTrue(np.isclose(non_batched[i].logits_BV[0], batched[i].logits_BV).all())

            for j in range(len(non_batched[i].tokenwise_log_probs)):
                nb_lps = non_batched[i].tokenwise_log_probs[j]
                b_lps = batched[i].tokenwise_log_probs[j]
                self.assertTrue(np.isclose(nb_lps[0], b_lps[0]).all())
                self.assertTrue(np.isclose(nb_lps[1], b_lps[1]).all())

            self.assertEqual(non_batched[i].continuations, batched[i].continuations)
            self.assertTrue(np.isclose(non_batched[i].acts[0], batched[i].acts).all())


if __name__ == "__main__":
    unittest.main()
