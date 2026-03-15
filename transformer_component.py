import torch
import torch.nn as nn
import torch.optim as optim

class Tranformer_com:

    def softmax(self,mat):
        mat = torch.exp(mat - torch.max(mat,axis=1,keepdim=True))
        return mat/torch.sum(mat,axis=1,keepdim=True)

    def masked_matrix(self,mat):
        for row in range(mat.shape[0]):
            for col in range(mat.shape[1]):
                if col > row:
                    mat[row][col] = -torch.inf
                else :
                    mat[row][col] = 0

        return mat 

    def masked_multi_head_attention(self,input_data,n_head):
        d_k = input_data.shape[1]


        if d_k%n_head == 0:
            W_o = torch.random.rand(d_k,d_k,required_grad=True)

            head_dict = {}
            list_Iq = []
            list_Ik = []
            list_Iv = []

            for head in range(n_head):
                Q = torch.random.rand(d_k,d_k//n_head,required_grad=True)
                K = torch.random.rand(d_k,d_k//n_head,required_grad = True)
                V = torch.random.rand(d_k,d_k//n_head,required_grad = True)
                list_Iq.append(Q),list_Ik.append(K),list_Iv.append(V)

            head_dict['Iq'] = list_Iq
            head_dict['Ik'] = list_Ik
            head_dict['Iv'] = list_Iv

            mask_input_list = []
            for head in range(n_head):
                Iq = torch.matmul(input_data,head_dict['Iq'][head])
                Ik = torch.matmul(input_data,head_dict['Ik'][head])
                Iv = torch.matmul(input_data,head_dict['Iv'][head])


                score = (Iq@Ik.T)/torch.sqrt(d_k//n_head)

                mat_for_mask = torch.ones(score.shape[0],score.shape[1])

                masked_score = score + self.masked_matrix(mat_for_mask)

                softmax_score = self.softmax(masked_score)

                mask_input_list.append(softmax_score@Iv)

                concat_head = torch.hstack(mask_input_list)

            return torch.matmul(concat_head,W_o)
        
        else :
            raise f'Please check you n_head ,n_head should completly divide the the column of input vector'
        
    def add_norm(self,input_data,masked_self_output):
        residual_add = masked_self_output + input_data
        return torch.layer_norm(residual_add)
    
class FeedForwardNN(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(FeedForwardNN,self).__init__()

        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out






                
