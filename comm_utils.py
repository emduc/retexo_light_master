"""Implement communication utilities for GNN models"""

from typing import Dict, List
from multiprocessing.pool import ThreadPool
import time

import os

import torch.distributed as dist
import torch
import torch.nn as nn
import numpy as np



from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class MultiThreadReducerCentralized:
    """Multi-threaded reducer for aggregating gradients in a centralized manner"""

    def __init__(self, model, device="cuda", sleep_time=0.1, pubkey=None, perf_store=None, measure_comm=False):
        self.model = model
        # self._handles = []
        # self._stream = None
        self._group = {}
        self.thread_pool = None
        self.sleep_time = sleep_time
        cnt = 0
        # for _, (name, param) in enumerate(self.model.named_parameters()):
        #     cnt+=1
        #     self._group[name] = dist.new_group()
        # self.thread_pool = ThreadPool(processes=cnt)
        # self._stream = torch.cuda.Stream(device=device)
        self.measure_comm = measure_comm
        self.comm_vol_store = perf_store
        self.pubkey = pubkey
        
        # TODO (for local tests)
        with open("private_key.pem", "rb") as private_file:
            self.private_key = serialization.load_pem_private_key(
                private_file.read(),
                password=None,
                backend=default_backend()
            )


    # def _reduce_encrypted(self, rank, world_size, model: nn.Module, shapes, num_elements, name, encrypted_grads_iv_key):
    #     def create_stream(shapes, num_elements, name, encrypted_grads_iv_key):
    #         self._stream.wait_stream(torch.cuda.current_stream())
    #         with torch.cuda.stream(self._stream):
    #             group = None#self._group[name]
    #             time.sleep(2 * self.sleep_time)
    #             encrypted_list = None
    #             if rank == 0:
    #                 encrypted_list = []
    #                 for _ in range(world_size):
    #                     # Placeholder tensor for gathering encrypted gradients
    #                     encrypted_list.append(torch.zeros_like(encrypted_grads_iv_key))

    #             # Gather encrypted gradients, IV, and keys
    #             dist.gather(encrypted_grads_iv_key, encrypted_list, dst=0, group=group)
                
    #             # Decrypt and aggregate the gradients
    #             if rank == 0:
    #                 decrypted_list = []
    #                 for encrypted in encrypted_list:
    #                     encrypted_grads_iv_key = encrypted.numpy()
    #                     iv = encrypted_grads_iv_key[-34:-32].tobytes()
    #                     encrypted_key = encrypted_grads_iv_key[-32:].tobytes()
    #                     encrypted_grads = encrypted_grads_iv_key[:-34] 
                        
    #                     symmetric_key = self.private_key.decrypt(
    #                         encrypted_key,
    #                         padding.OAEP(
    #                             mgf=padding.MGF1(algorithm=hashes.SHA256()),
    #                             algorithm=hashes.SHA256(),
    #                             label=None
    #                         )
    #                     )
    #                     # Decrypt gradients
    #                     cipher = Cipher(algorithms.AES(symmetric_key), modes.CFB(iv), backend=default_backend())
    #                     decryptor = cipher.decryptor()
    #                     decrypted_bytes = decryptor.update(encrypted_grads.tobytes()) + decryptor.finalize()
    #                     decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.float32)
    #                     decrypted_list.append(decrypted_array)

    #                 # Aggregate decrypted gradients
    #                 aggregated_grads = np.sum(decrypted_list, axis=0)

    #                 # Convert aggregated gradients back to tensor and set to param.grad
    #                 start = 0
    #                 i = 0
    #                 for param in model.parameters():
    #                     if param.grad is None:
    #                         continue
    #                     # num_elem = param.grad.numel()
    #                     grad_array = aggregated_grads[start:start + num_elements[i]].reshape(shapes[i])
    #                     param.grad.copy_(torch.tensor(grad_array).view_as(param.grad))
    #                     start += num_elements[i]
    #                     i+=1

    #             # Broadcast the aggregated gradients from rank 0 to all workers
    #             for _, param in enumerate(model.parameters()):
    #                 if param.grad is not None:
    #                     dist.broadcast(param.grad, src=0, group=group)

    #             if self.measure_comm:
    #                 # Add communication volume for gradient reduction
    #                 if rank == 0:
    #                     cv = 2 * get_comm_size_param(param.grad) * (world_size - 1)
    #                 else:
    #                     cv = 2 * get_comm_size_param(param.grad)
    #                 self.comm_vol_store.add_cv_grad_reduce_t(cv)

    #     self._handles.append(self.thread_pool.apply_async(create_stream, (shapes, num_elements, name, encrypted_grads_iv_key)))

    #     # torch.cuda.current_stream().wait_stream(self._stream)
        
    def master_aggregate_gradients(self, cfg, layer):
        """Aggregate the gradients on the master node"""
        world_size = cfg.num_partitions + 1
        
        all_grads = []
        num_elements = []
        shapes = []
        
        dummy_iv = os.urandom(16)
        dummy_key = os.urandom(32)
    
        # with torch.no_grad():
        for _, (name, param) in enumerate(self.model.named_parameters()):
            all_grads.append(torch.zeros_like(param).cpu().numpy())
            shapes.append(param.shape)
            num_elements.append(param.numel())
    
        all_grads_array = np.concatenate([grad.flatten() for grad in all_grads])
        target_tensor = torch.tensor(np.frombuffer(all_grads_array.tobytes() + dummy_iv + dummy_key, dtype=np.float32))
        
        for round in range(cfg.num_rounds[layer]):
            encrypted_list = []
            for _ in range(1, world_size):  # Exclude master itself
                # Placeholder tensor for gathering encrypted gradients
                encrypted_grads_iv_key = torch.zeros(target_tensor.shape[0], dtype=torch.float32)  
                
                # Adjust the size as needed
                dist.recv(encrypted_grads_iv_key, src=_, tag=round)
                encrypted_list.append(encrypted_grads_iv_key.numpy())
              
            decrypted_list = []
            for encrypted in encrypted_list:
                # Extract metadata
                num_params = int.from_bytes(encrypted[:1], 'little')
                metadata_size = 1 + num_params# * 4
                indexes = np.frombuffer(encrypted[1:metadata_size], dtype=np.int32)
                
                gradients_size = 0
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    if i in indexes:
                        gradients_size += param.numel()
                
                offset = metadata_size + gradients_size
                iv = encrypted[offset:offset+4].tobytes()
                encrypted_key = encrypted[offset+4:offset+68].tobytes()
                encrypted_grads = encrypted[metadata_size:offset]

                symmetric_key = self.private_key.decrypt(
                    encrypted_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                # Decrypt gradients
                cipher = Cipher(algorithms.AES(symmetric_key), modes.CFB(iv), backend=default_backend())
                decryptor = cipher.decryptor()
                decrypted_bytes = decryptor.update(encrypted_grads.tobytes()) + decryptor.finalize()
                decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.float32)
                decrypted_list.append(decrypted_array)

            # Aggregate decrypted gradients
            aggregated_grads = np.sum(decrypted_list, axis=0)

            # Convert aggregated gradients back to tensor and set to param.grad
            start = 0
            # i = 0
            index = 0
            for param in self.model.parameters():
                if index in indexes:
                    grad_array = aggregated_grads[start:start + param.numel()].reshape(param.shape)
                    param.grad = torch.tensor(grad_array).view(param.shape)
                    start += param.numel()
                    # i += 1
                index+=1


            # Broadcast the aggregated gradients from rank 0 to all workers
            for (name, param) in self.model.named_parameters():
                if param.grad is not None:
                    dist.broadcast(param.grad, src=0)

def extract_node_type(name, node_types):
    tokens = name.split(".")
    for type in node_types:
        if type in tokens:
            return type
    return "all"

def aggregate_metrics(metrics: Dict):
    """Aggregate the metrics across workers

    Parameters
    ----------
    metrics : Dict
        Metrics to aggregate

    Returns
    -------
    Dict
        Aggregated metrics
    """

    for k, v in metrics.items():
        t = torch.tensor(v).cuda()
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=False)
        metrics[k] = t.item() 

    return metrics

def sync_model(model: nn.Module):
    """Sync the model across workers

    Parameters
    ----------
    model : nn.Module
        Model to sync

    Returns
    -------
    nn.Module
        Synced model
    """

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

def get_comm_size_param(param):
    """Get the communication size of a parameter

    Parameters
    ----------
    param : torch.Tensor
        Parameter

    Returns
    -------
    int
        Communication size of the parameter
    """

    return param.numel() * param.element_size()
