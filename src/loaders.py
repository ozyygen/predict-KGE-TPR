import  torch
import  torch.nn as nn
from    torch.utils.data import DataLoader, Dataset

import random

class TestDataset(Dataset):

  def __init__(self,edges,num_nodes,num_rels=1,filter=True,mode='tail'):

    self.edges_index  = edges    
    self.num_rels     = num_rels
    self.num_nodes    = num_nodes  
    self.num_edges    = len(edges)
    self.edges_dict   = {}
    self.filter       = filter
    self.mode         = mode
    
    # create a dict (for neg sample filtering)
    for i in range(self.num_edges):
      h = self.edges_index[i][0]
      t = self.edges_index[i][2]
      r = self.edges_index[i][1]
      if (h,t) not in self.edges_dict:
        self.edges_dict[(h,t)] = []
      self.edges_dict[(h,t)].append(r)

  def __len__(self):
      return self.num_edges

  def _sample_negative_edge(self,idx,max_num=100,mode='tail'):

      num_neg_samples = 0      
      triplets        = []      
      nodes           = list(range(self.num_nodes))
      random.shuffle(nodes)
      r               = self.edges_index[idx][1]

      while num_neg_samples < max_num:
                
        if mode == 'tail':
          t   = nodes[num_neg_samples]                 
          h   = self.edges_index[idx][0]
        else:
          t   = self.edges_index[idx][2]                  
          h   = nodes[num_neg_samples]                
        ht = torch.tensor([h,t]) 
                  
        if not self.filter:
          triplets.append([ht,r])
        else:
          if (h,t) not in self.edges_dict:
            triplets.append([ht,r])

          elif r not in self.edges_dict[(h,t)]:
            triplets.append([ht,r])

        num_neg_samples+=1
        if num_neg_samples == len(nodes):
          break

      return triplets

  def __getitem__(self,idx):

      pos_samples  = [torch.tensor([self.edges_index[idx][0],
                                    self.edges_index[idx][2]]),
                      self.edges_index[idx][1]]
      neg_samples  = self._sample_negative_edge(idx,mode=self.mode)      
      tuples100     = torch.stack([pos_samples[0]]+[ht for ht,_ in neg_samples])
      edges100      = torch.stack([torch.tensor(pos_samples[1])] + [torch.tensor(r) for _,r in neg_samples]) 
      return tuples100, edges100

class TrainDataset(Dataset):

  def __init__(self,edges,quads,num_nodes,num_rels=1,filter=True):

    self.edges_index  = edges    
    self.num_rels     = num_rels
    self.num_nodes    = num_nodes  
    self.num_edges    = len(edges)
    self.edges_dict   = {}
    self.filter       = filter
    self.quads_index  = quads
    self.quads_dict   = {}

    self.quads_count = 0
    
    # create a dict (for neg sampling)
    for i in range(self.num_edges):
      h = self.edges_index[i][0]
      t = self.edges_index[i][2]
      r = self.edges_index[i][1]
      ht = (h,t)
      if ht  not in self.edges_dict:
        self.edges_dict[ht] = []
      self.edges_dict[ht].append(r)

    # creat quadruple dict
    for i in range(len(self.quads_index)):
      h = self.quads_index[i][0]
      t = self.quads_index[i][4]

      l1= self.quads_index[i][1]
      l2= self.quads_index[i][3]    

      if (h,t) in self.edges_dict.keys():
        self.quads_count+=1    
      
      if (h,t) not in self.quads_dict:
        self.quads_dict[(h,t)] = []
      self.quads_dict[(h,t)].append((l1,l2))

  def __len__(self):
      return self.num_edges

  def _sample_quadruples(self,idx):
      
      h   = self.edges_index[idx][0]
      t   = self.edges_index[idx][2]
      ht = (h,t)
    
      if ht in self.quads_dict:         
        return self.quads_dict[ht][0]
      else:
        return (-1,-1)
      
  def _sample_negative_edge(self,idx):

      sample  = random.uniform(0,1)
      found   = False

      while not found:
        if sample <= 0.4: # corrupt head
          h   = torch.randint(0,self.num_nodes,(1,))
          t   = self.edges_index[idx][2]
          r   = self.edges_index[idx][1]
        elif 0.4 < sample <= 0.8: # corrupt tail
          t   = torch.randint(0,self.num_nodes,(1,))
          h   = self.edges_index[idx][0]
          r   = self.edges_index[idx][1]
        else: # corrupt relation          
          r   = torch.randint(0,self.num_rels,(1,))[0]
          h   = self.edges_index[idx][0]
          t   = self.edges_index[idx][2]
        
        if not self.filter:
          found = True
        else:
          if (h,t) not in self.edges_dict:
            found = True
          elif r not in self.edges_dict[(h,t)]:
            found = True

      return [torch.tensor([h,t]),r]

  def __getitem__(self,idx):

      pos_sample  = [[torch.tensor(self.edges_index[idx][0]),
                      torch.tensor(self.edges_index[idx][2])],
                     self.edges_index[idx][1]]
      neg_sample  = self._sample_negative_edge(idx)  
      quad_sample = self._sample_quadruples(idx)          
      return pos_sample, neg_sample,quad_sample
