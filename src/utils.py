import  os
import  re
import  pandas as pd 
import  numpy as np
#obonet package for reading .obo extensions - Gene Ontology
import obonet
import random
# Converts aspect (F,P,or C) to relation (GO_0003674,GO_0008150,or GO_0005575), 
# pre-fixed with the PURL 
PURL = "http://purl.obolibrary.org/obo/"

def return_rel_type(date,qualifier, aspects):

      rel = ""

      #qualifier for gaf files before 2021 
      if int(date) < 2021:
        str_q   = str(qualifier)
        aspect  = str(aspects)          
      #qualifier for gaf files >= 2021
      else:
        str_q   = str(qualifier).split('|')[0]
        aspect  = str(aspects)

      if str_q != 'NOT':
        #molecular function
        if    aspect == 'F':
          rel = PURL + 'GO_0003674'
        #biological process
        elif  aspect == 'P':
          rel = PURL + 'GO_0008150'
        #cellular component
        elif  aspect == 'C':
          rel = PURL + 'GO_0005575'
      else:
        rel = "NOT"

      return rel

# Corresponding gene ontology version that will be included in training set,
# GO version file-date should be the same with the first version date 

def get_subsumption_hierarchy(first_version,node,graph_go):
  
    data = list()

    #Find edges to parent terms,
    # given child (= node)
    for child, parent, key  in  graph_go.out_edges(node, keys=True):  
        
        if key == "is_a":
          data.append((PURL + str(child), 
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" , 
                    PURL + str(parent)))     
        elif key == "part_of":
          data.append((PURL + str(child) , 
                    PURL + "BFO_0000050" , 
                    PURL + str(parent) ))
    

    # create Dataframe 
    final_data = pd.DataFrame()
    final_data = final_data.from_records(data) 
    
    return final_data
   

def read_gene_ontology(onto_path):

    graph_go = obonet.read_obo(onto_path+".obo")
    return graph_go

def read_goa_file(goa_path):
 
  
  df = pd.read_csv(goa_path+".gaf.gz",
                      compression='gzip', comment='!', nrows = 20000,
                      header=None, usecols=[1,3,4,8,13], 
                      names=["db_object_ID","qualifier","GO_ID", "aspect","date"], 
                      delimiter="\t")

  return df

#returns specific version of GOA & related GO terms in GO
def generate_goa_graph(ONTO_GDRIVE_PATH, GOA_DRIVE_PATH, spec_version, v_number):

  data  = list()
  df_dc = dict()
  go_df     = pd.DataFrame()
  final_df  = pd.DataFrame(columns=("head","relation","tail"))

  df_dc     = read_goa_file(GOA_DRIVE_PATH+spec_version)
  
  if(v_number == 1):
    graph_go  = read_gene_ontology(ONTO_GDRIVE_PATH+spec_version)

  for i,row in df_dc.iterrows():
    
    date = str(row["date"])
    year = date[:4]

    rel = return_rel_type(year,row["qualifier"],row["aspect"])
    
    if (rel != "NOT"):
      #for test and valid sets
      if(v_number == 0):
          data.append(("https://www.uniprot.org/uniprot/" + row["db_object_ID"] , 
                       rel , 
                       PURL+row["GO_ID"] ))

      #for train set (+ GO semantic )  
      else:       
          data.append(("https://www.uniprot.org/uniprot/" + row["db_object_ID"] , 
                       rel , 
                       PURL+row["GO_ID"] ))
          
          #GO semantic from tail go term
          node    = row["GO_ID"]
          temp_go = get_subsumption_hierarchy(spec_version,node,graph_go)
          go_df   = go_df.append(temp_go,ignore_index=True)
      
   
  
  # structure: subject -> gene ,product p -> rel , object -> gene function
  final_df = pd.DataFrame.from_records(data)
  final_df = final_df.append(go_df,ignore_index=True)

  return final_df#returns specific version of GOA & related GO terms in GO

def get_entities(triple_df):
  entities = set()
  for i, (head, relation, tail) in triple_df.iterrows():

    entities.add(head)
    entities.add(tail)
  return entities

def generate_test_valid(triple_df, second_df, entities_v1):
  #V2 triple difference from V1
  diffv2_v1 = second_df.merge(
    triple_df, how='outer', indicator=True
).query('_merge == "left_only"').drop('_merge', 1)

  print("V2 different triple count: " , len(diffv2_v1))

  #existing entities in changed triples in V2
  diffv2_v1_ex = set()
  for i, (head, relation, tail) in diffv2_v1.iterrows():

    if (head in entities_v1) and (tail in entities_v1)  :
        diffv2_v1_ex.add((head, relation, tail))

  print("different triples with existed entities" , len(diffv2_v1_ex))
  return diffv2_v1_ex

def split_test_valid(diffv2_v1_ex): 
  #split changed triples with existed entities in V2 into two
  #test and valid sets
  #1/2 of data to be test and 1/2 valid
  #randomizing 
  my_list = list(diffv2_v1_ex)
  my_list = random.shuffle(my_list)
  diffv2_v1_ex = set(my_list)
 
  slen = round(len(diffv2_v1_ex) / 2) # we need 2 subsets
  test = set()
  valid = set()
  entity_test =set()
  entity_valid =set()
  rel_test = set()
  rel_valid = set()

  while not len(diffv2_v1_ex) == slen:
    v = diffv2_v1_ex.pop()
    valid.add(v)
  
  while len(diffv2_v1_ex) >= 1:
    v = diffv2_v1_ex.pop()
    test.add(v)
    
  return test,valid

def txt_gen(file_name,data):
  data_l = list()
  data_df = pd.DataFrame()
  with open(file_name, 'w') as f:
    for item in data: 
      f.write(item[0]+"\t"+item[1]+"\t"+item[2]+"\n")
      data_l.append((item[0],item[1],item[2]))
  data_df = data_df.from_records(data_l)
  return data_df  

def add_level1_semantic(graph_go, triple_df,second_df,spec_version_v1,entities_v1):
  test_valid_data = set()
  diffv2_v1 = second_df.merge(
    triple_df, how='outer', indicator=True
).query('_merge == "left_only"').drop('_merge', 1)
  #add into test set : changed triples with existed entities in V2
  for i, (head, relation, tail) in diffv2_v1.iterrows():
    if (head in entities_v1) and (tail in entities_v1)  :
      test_valid_data.add((head, relation, tail))
    # Find edges to parent terms
    go_term = (str(tail)).split('/')[4]

    #SAMPLE: 
      #annotation : x F G ;  G: gene function
      #gene ontology: G key D ; G = child,  key= "subClassOf"
      #result: x F D

    for child, parent, key in graph_go.out_edges(go_term, keys=True):
      #connect implicit annotations to the same head and rel pair as objects
        
      test_valid_data.add((head,relation,PURL + parent))
  return test_valid_data

def add_level2_semantic(graph_go, diffv2_v1, entities_v1, spec_version_v1):
  #Level 2: Use levels as hyperparameters such as 1,2,3..
  #add level 1 and 2 hierarchical implicit annotations into test set
  test_valid_data = set()
  #add into test set : changed triples with existed entities

  for i, (head, relation, tail) in diffv2_v1.iterrows():

  #existing entities in changed triples in V2
    if (head in entities_v1) and (tail in entities_v1)  :
      test_valid_data.add((head, relation, tail))
    
  # Find edges to parent terms
    go_term = tail.split('/')[4]

    #SAMPLE:
    #annotation : x F G ;  G: gene function
    #gene ontology: G key D ; G = child,  key= "subClassOf"
    #result: x F D

  # first level of hierarchical semantic
    for child, parent, key in graph_go.out_edges(go_term, keys=True):
      #connect implicit annotations to the same head and rel pair as objects
      test_valid_data.add((head,relation,PURL + parent))
      #second level of hierarchical semantic
      for child_p, parent_p, key_p in graph_go.out_edges(parent, keys=True):
          test_valid_data.add((head,relation,PURL + parent_p))
  return test_valid_data

def final_test_valid_sc2(triple_df, test_valid_data,graph_go):
  # implicit annotations that can be captured in the train set
  temp_set = set()
  for i, (head, relation, tail) in triple_df.iterrows():
    go_term = tail
    for child, parent, key in graph_go.out_edges(go_term, keys=True):
      #connect implicit annotations to the same head and rel pair as objects
      temp_set.add((head,relation,PURL + parent))
  temp_set.update(test_valid_data)
  return temp_set