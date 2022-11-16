import  torch
import  torch.nn as nn
import  torch.nn.functional as F  
from    torch.utils.data import DataLoader

import  numpy as np

from .loaders import TestDataset,TrainDataset

class BaseModel(nn.Module):
    
    def __init__(self,num_entities,num_relations,emb_dim):
        super(BaseModel,self).__init__()
        self.entity_embds = nn.Parameter(torch.randn(num_entities,emb_dim))
        self.rel_embds    = nn.Parameter(torch.randn(num_relations,emb_dim))
        self.rel_dim = emb_dim
            
    def forward(self):        
        self.entity_embds.data[:-1,:].div_(self.entity_embds.data[:-1,:].norm(p=2,dim=1,keepdim=True))
        return self.entity_embds, self.rel_embds
    
    def _train(self,train_trips,train_quads,train_batch_size=26,num_epoches=150):

        train_db = TrainDataset(train_trips,train_quads,self.num_entities,filter=False)
        train_dl = DataLoader(train_db,batch_size=train_batch_size,shuffle=True)

        train_losses = []
        for e in range(num_epoches):  
    
            self.model.train()
            losses = []
            
            try:
                for batch in train_dl:
                    pos_samples , neg_samples, quads = batch
                    
                    self.optimizer.zero_grad()

                    loss = self.compute_loss(pos_samples, neg_samples, quads)
                    
                    loss.backward()
                    self.optimizer.step()
                    if np.isnan(loss.item()):
                        ...
                    else:
                        losses.append(loss.item())
                    
            except TypeError:
                pass

            if e % 50 == 0:
                if len(losses)!=0:
                    mean_loss = np.array(losses).mean()
                    print('epoch {},\t train loss {:0.02f}'.format(e,mean_loss))
                else:
                    mean_loss = np.NaN
                
            train_losses.append(mean_loss)

        return train_losses
    
    
    def _eval(self,test_edges_index):
        
        self.model.eval() 
  
        test_db = TestDataset(test_edges_index,self.num_entities,filter=False)
        test_dl = DataLoader(test_db,batch_size=len(test_db),shuffle=True)

        hits1_list  = []
        hits10_list = []
        MR_list     = []
        MRR_list    = []

        e_embeds, r_embeds = self.model()

        batch = next(iter(test_dl))
        edges, edge_rels = batch
        batch_size, num_samples, _ = edges.size()
        edges = edges.view(batch_size*num_samples,-1)
        edge_rels = edge_rels.view(batch_size*num_samples,-1)

        h_embeds = torch.index_select(e_embeds,0,torch.tensor([int(x) for x in edges[:,0]]))
        r_embeds = torch.index_select(r_embeds,0,torch.tensor([int(x) for x in edge_rels.squeeze()]))
        t_embeds = torch.index_select(e_embeds,0,torch.tensor([int(x) for x in edges[:,1]]))

        scores = torch.norm(h_embeds+r_embeds-t_embeds,p=1,dim=1).view(batch_size,num_samples)

        # sort and calculate scores
        argsort = torch.argsort(scores,dim = 1,descending= False)
        rank_list = torch.nonzero(argsort==0,as_tuple=False)
        rank_list = rank_list[:,1] + 1
  
        hits1_list.append( (rank_list <= 1).to(torch.float).mean() )
        hits10_list.append( (rank_list <= 10).to(torch.float).mean() )
        MR_list.append(rank_list.to(torch.float).mean())
        MRR_list.append( (1./rank_list.to(torch.float)).mean() )
  
        hits1 = sum(hits1_list)/len(hits1_list)
        hits10 = sum(hits10_list)/len(hits10_list)
        mr = sum(MR_list)/len(MR_list)
        mrr = sum(MRR_list)/len(MRR_list)

        print('hits@1 ',hits1,',hits@10 ',hits10,',MR ',mr,',MRR ',mrr)
    

class TransE(BaseModel):
    
    def __init__(self,num_entities,num_relations,emb_dim=30, type_count=1,lr=1e-3):
        super(TransE,self).__init__(num_entities,num_relations,emb_dim)
        
        self.num_entities = num_entities
        
        self.model        = BaseModel(num_entities,num_relations,emb_dim)
        self.optimizer    = torch.optim.Adam(self.model.parameters(),lr=lr)
    
    def compute_loss(self,pos_edges,neg_edges,quads):
        return self.TransE_loss(pos_edges[0],neg_edges[0],pos_edges[1],neg_edges[1])

    def TransE_loss(self,pos_edges,neg_edges,pos_rel,neg_rel):
        """ Ensures tail embeding and translated head embeding are nearest neighbour """
        
        entity_embds, rel_embds = self.model()
  
        pos_h_embs = torch.index_select(entity_embds,0,torch.tensor([int(x) for x in pos_edges[0]]))
        pos_t_embs = torch.index_select(entity_embds,0,torch.tensor([int(x) for x in pos_edges[1]]))
        pos_r_embs = torch.index_select(rel_embds,0,torch.tensor([int(x) for x in pos_rel]))  

        neg_h_embs = torch.index_select(entity_embds,0,torch.tensor([int(x) for x in neg_edges[:,0]]))
        neg_t_embs = torch.index_select(entity_embds,0,torch.tensor([int(x) for x in neg_edges[:,1]]))
        neg_r_embs = torch.index_select(rel_embds,0,torch.tensor([int(x) for x in neg_rel]))
  
        d_pos = torch.norm(pos_h_embs + pos_r_embs - pos_t_embs, p=1, dim=1)
        d_neg = torch.norm(neg_h_embs + neg_r_embs - pos_t_embs, p=1, dim=1)
        ones  = torch.ones(d_pos.size(0))

        margin_loss = torch.nn.MarginRankingLoss(margin=1.)
        loss        = margin_loss(d_neg,d_pos,ones)

        return loss

class rTransE(TransE):
    
    def __init__(self,num_entities,num_relations,emb_dim=30, type_count=1,lr=1e-3):
        super(rTransE,self).__init__(num_entities,num_relations,emb_dim=emb_dim, type_count=type_count,lr=lr)
        
    
    def compute_loss(self,pos_edges,neg_edges,quads):
        return self.TransE_loss(pos_edges[0],neg_edges[0],pos_edges[1],neg_edges[1]) \
                                    + self.rTransE_loss(pos_edges[0],quads[0],quads[1])
    
    def rTransE_loss(self,ht_edges,l1_rel,l2_rel):
        """ Ensures  distance is small for composition rule """
        
        entity_embds, rel_embds = self.model()

        idx = [i for i,x in enumerate(l1_rel) if x!=-1]
  
        if len(idx) == 0 :
            return torch.tensor(0)

        h_embs = torch.index_select(entity_embds,0,torch.tensor([int(ht_edges[0][i]) for i in idx]))
        t_embs = torch.index_select(entity_embds,0,torch.tensor([int(ht_edges[1][i]) for i in idx]))  

        l1_embs = torch.index_select(rel_embds,0,torch.tensor([int(l1_rel[i]) for i in idx]))
        l2_embs = torch.index_select(rel_embds,0,torch.tensor([int(l2_rel[i]) for i in idx]))

        d_pos = torch.norm(h_embs + l1_embs + l2_embs - t_embs, p=2, dim=1)
  
        return d_pos.mean()    

