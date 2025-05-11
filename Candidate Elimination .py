#!/usr/bin/env python
# coding: utf-8

# In[10]:

data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "no"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "yes"]
]

def candidate_elimination(data):
    S=["#"]*(len(data[0])-1)
    G=[["?"]*(len(data[0])-1)]
    
    for r in data:
        attribute = r[:-1]
        label= r[-1]
    
        if label=="yes":
            for i in range(len(S)):
                if S[i] == "#":
                    S[i] = attribute[i]
                elif S[i] != attribute[i]:
                    S[i] = "?"
                    
            new_G=[]
            for g in G:
                valid = True
                for i in range(len(S)):
                    if g[i]!="?" and g[i]!=S[i]:
                        valid = False
                        break
                if valid:
                    new_G.append(g)
            G=new_G
            
        elif label=="no":
            new_G=[]
            for g in G:
                for i in range(len(g)):
                    if g[i]=="?":
                        new_hypo=g[:] #copy same content
                        new_hypo[i]=attribute[i] #Replace "?" with specific attribute
                        new_G.append(new_hypo)
            
            G=new_G

    return S,G
                
Specific_Hypothesis, General_Hypothesis = candidate_elimination(data)

print("Specific_Hypothesis: ",Specific_Hypothesis)
print("General Hypothesis: ",General_Hypothesis)
                
            


# In[ ]:




