import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class BiggerConvInputModel(nn.Module):
    def __init__(self):
        super(BiggerConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(256)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x



class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

  
class FCOutputModel(nn.Module): 
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256) # optimal par of pixel
        self.fc3 = nn.Linear(256, 10)
        # self.fc2 = nn.Linear(1000, 500)
        # self.fc3 = nn.Linear(500, 100)
        # self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss
        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')

        self.conv = ConvInputModel()
        
        self.relation_type = args.relation_type
        
        
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((24+2)*2+18, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2)
        
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1, 25, 1)
        qst = torch.unsqueeze(qst, 2)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26+18)
        x_i = x_i.repeat(1, 25, 1, 1)  # (64x25x25x26+18)
        x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26+18)
        x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, 25, 1)  # (64x25x25x26+18)
        
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+18)
    
        # reshape for passing through network
        x_ = x_full.view(mb * (d * d) * (d * d), 70)  # (64*25*25x2*26*18) = (40.000, 70)
            
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        
        x_g = x_.view(mb, (d * d) * (d * d), 256)

        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)


class BiggerRN(BasicModel):
    def __init__(self, args):
        super(BiggerRN, self).__init__(args, 'BiggerRN')
        
        
        self.conv = BiggerConvInputModel()
        
        self.relation_type = args.relation_type
        
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((256+2)*2+11, 2000)

        self.g_fc2 = nn.Linear(2000, 2000)
        self.g_fc3 = nn.Linear(2000, 2000)
        self.g_fc4 = nn.Linear(2000, 2000)

        self.f_fc1 = nn.Linear(2000, 1000) # after layer 2 will be called by FCOutputModel function
        self.f_fc2 = nn.Linear(1000, 500) # delete
        self.f_fc3 = nn.Linear(500, 100)
        self.f_fc4 = nn.Linear(100, 10)
        # self.f_fc5 = nn.Linear(100, 10)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        # self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 256 x 5 x 5)
        
        """g"""
        mb = x.size()[0] # 64
        n_channels = x.size()[1] # 256
        d = x.size()[2] # 5
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1) # x_flat = (64 x 25 x 256)
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2) #(64 x 25 x (256 + 2))
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1) #(64,11) -> (64,1,11)
        qst = qst.repeat(1, 25, 1) # 64 x 25 x 11
        qst = torch.unsqueeze(qst, 2) #64 x 25 x 1 x 11

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # 64 x 1 x 25 x (256 + 2)
        x_i = x_i.repeat(1, 25, 1, 1)  # 64 x 25 x 25 x (256 + 2)
        x_j = torch.unsqueeze(x_flat, 2)  #  64 x 25 x 1 x (256 + 2)
        x_j = torch.cat([x_j, qst], 3) # 64 x 25 x 1 x (256 + 2 + 11)
        x_j = x_j.repeat(1, 1, 25, 1)  #  64 x 25 x 25 x (256 + 2 + 11)
        
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # 64 x 25 x 25 x ((256 + 2) + (256 + 2 + 11))
    
        # reshape for passing through network
        x_ = x_full.view(mb * (d * d) * (d * d), 2*(256 + 2) + 11)  # (64*25*25x2*258+11) = (40.000, 527)
            
        x_ = self.g_fc1(x_) # 64*25*25 x 2000
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_) # 64*25*25 x 2000
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_) # 64*25*25 x 2000
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_) # 64*25*25 x 2000
        x_ = F.relu(x_)
        
        # reshape again and sum
        
        x_g = x_.view(mb, (d * d) * (d * d), 2000)

        x_g = x_g.sum(1).squeeze() # 64 x 2000
        
        """f"""
        x_f = self.f_fc1(x_g) # 64 x 2000
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f) # 64 x 1000
        x_f = F.relu(x_f)
        x_f = self.f_fc3(x_f) # 64 x 500
        x_f = F.relu(x_f)
        x_f = self.f_fc4(x_f) # 64 x 100
        # x_f = F.relu(x_f)
        # x_f = self.f_fc5(x_f) # 64 x 10
        x_f = F.log_softmax(x_f, dim=1) # 64 x 10
        
        # return self.fcout(x_f) #after 2 layers
        return x_f #after 2 layers


class StateRN(BasicModel):
    def __init__(self, args):
        super(StateRN, self).__init__(args, 'StateRN')

        self.relation_type = args.relation_type


        '''
        FOR IMG DESCRIPTION INPUT
        '''
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((4+2)*2+11, 512)
        #4 MLP layers 512 per layer
        self.g_fc2 = nn.Linear(512, 512)
        self.g_fc3 = nn.Linear(512, 512)
        self.g_fc4 = nn.Linear(512, 512)

        # A three-layer MLP consisting of
        # 512, 1024 (with 2% dropout) and 29 units with ƒReLU non-linearities
        # was used for fθ.
        # self.f_fc1 = nn.Linear(512, 1024)
        # self.f_fc2 = nn.Linear(512, 1024)
        # self.f_fc3 = nn.Linear(512, 1024)
        # self.f_fc4 = nn.Linear(64, 10)
        
        # f function: 3 layers
        self.f_fc1 = nn.Linear(512, 1024) #oposit?
        self.f_fc2 = nn.Linear(512, 1024)
        self.f_fc3 = nn.Linear(64, 10)
        # (?) #final output length depends on the answer embedding (10 in this case?)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i / 5 - 2) / 2., (i % 5 - 2) / 2.]

        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:, i, :] = np.array(cvt_coord(i))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, img_description, qst):
        # 4 columns,corresponds to object size (5x5 in conv)
        # 6 objects, corresponds to img size = 256 pixels (?)
        # 64 = minibatch size
        # x = self.conv(img)  ## x = (64 x 24 x 5 x 5)
        x = img_description # x = (64 x 6 x 4)
        # x.numpy()
        # x = x.T
        # #obj should only have 3D
        # print(len(x[0]))
        # print('------')
        # print(len(x[1]))
        
        
        """g"""
        mb = x.size()[0]
        # mb = x.size()[0]
        n_channels = x.size()[1]
        # n_channels = x.size()[1]
        # print(n_channels)
        # d1 = x.size()[2]
        # d2 = 1
        # d = x.size()[0]
        d = x.size()[2]
        # print(d)

        # x_flat = x
        x_flat = x.view(mb, n_channels, d * d).permute(0, 2, 1) 
        # x_flat = x.view(mb, n_channels, d * d)
        # print("coord_tensor {}".format(len(self.coord_tensor)))
        
        
        # x_flat = torch.cat([x_flat, self.coord_tensor],2)
        # print("x_flat size: {}".format(len(x_flat[0])))

        # x_flat = x
        #4 dimensions?

        # add coordinates
        # coord = torch.unsqueeze(self.coord_tensor, 2)  #  64 x 25 x 1 x (256 + 2)
        # coord = coord.repeat(1, 1, 25, 1)  #  64 x 25 x 25 x (256 + 2 + 11)
        # x_flat = torch.cat([x_flat, coord], 2)


        # add question everywhere
        qst = torch.unsqueeze(qst, 1)  # (64,11) -> (64,1,11)
        # print("QST1 {}".format(len(qst)))
        qst = qst.repeat(1, 3, 1)  # 64 x 4 x 11
        # qst = qst.repeat(1, 9, 1)  # 64 x 4 x 11
        # qst = qst.repeat(1, 25, 1)  # 64 x 4 x 11
        # print("QST2 {}".format(len(qst)))
        qst = torch.unsqueeze(qst, 2)  # 64 x 4 x 1 x 11
        print("QST {}".format(len(qst)))
        
        
        
        x_i = torch.unsqueeze(x_flat, 1)  # 64 x 1 x 25 x (256 + 2)
        # print(len(x_i))
        # x_i = x_i.repeat(1, 25, 1, 1)  # 64 x 25 x 25 x (256 + 2)
        x_i = x_i.repeat(1, 6, 1, 1)  # 64 x 25 x 25 x (256 + 2)
        
        x_j = torch.unsqueeze(x_flat, 2)  #  64 x 25 x 1 x (256 + 2)
        x_j = torch.cat([x_j, qst], 3) # 64 x 25 x 1 x (256 + 2 + 11)
        # x_j = x_j.repeat(1, 1, 25, 1)  #  64 x 25 x 25 x (256 + 2 + 11)
        x_j = x_j.repeat(1, 1, 3, 1)  #  64 x 25 x 25 x (256 + 2 + 11)

        # cast all pairs against each other
        # x_flat: 64 x 24 x 6
        # x_i = torch.unsqueeze(x_flat, 1)  # 64 x 1 x 4 x 6
        # print(x_i[0])
        # x_i = x_i.repeat(1, 6, 1, 1)  # 64 x 4 x 4 x 6
        # print("size of x_i{}".format(len(x_i[2])))
        
        # x_j = torch.unsqueeze(x_flat, 2)  # 64 x 4 x 1 x 6
        # x_j = torch.cat([x_j,qst], 3)  # 64 x 4 x 1 x (6 + 11)
        # x_j = x_j.repeat(1, 1, 3, 1)  # 64 x 4 x 4 x (6 + 11)

        # concatenate all together
        print("size of x_i {}".format(len(x_i[2])))
        print("size of x_j {}".format(len(x_j[2])))
        x_full = torch.cat([x_i,x_j], 3)  # 64 x 4 x 4 x ((6) + (6 + 11))

        # reshape for passing through network
        x_ = x_full.view(mb * (d * d) * (d * d), 2 * 6 + 11)  # (64*4*4 x (2*6+11)) = (1024, 23)

        x_ = self.g_fc1(x_)  # 64*4*4 x 512
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)  # 64*4*4 x 512
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)  # 64*4*4 x 512
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)  # 64*4*4 x 512
        x_ = F.relu(x_)

        # reshape again and sum
        x_g = x_.view(mb, (d * d) * (d * d), 512)

        x_g = x_g.sum(1).squeeze()  # 64 x 512

        """f"""
        # unsure of these dimensions
        x_f = self.f_fc1(x_g)  # 64 x 1024
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)  # 64 x 1024
        x_f = F.relu(x_f)
        x_f = F.dropout(x_f) # 2%
        # nn.Dropout(0.02)
        x_f = self.f_fc3(x_f)  # 64 x 1024
        x_f = F.log_softmax(x_f, dim=1)  # 64 x 10

        return x_f
    
    

class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv  = ConvInputModel()
        self.fc1   = nn.Linear(5*5*24 + 18, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        #print([ a for a in self.parameters() ] )
  
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)
        
        x_ = torch.cat((x, qst), 1)  # Concat question
        
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        
        return self.fcout(x_)


#
# C CLEVR from state descriptions
# The model that we train on the state description version of CLEVR is similar to the model trained
# on the pixel version of CLEVR, but without the vision processing module. We used a 256 unit LSTM
# for question processing and word-lookup embeddings of size 32. For the RN we used a four-layer
# MLP with 512 units per layer, with ReLU non-linearities for gθ. A three-layer MLP consisting of
# 512, 1024 (with 2% dropout) and 29 units with ReLU non-linearities was used for fθ. To train the
# model we used 10 distributed workers that synchronously updated a central parameter server. Each
# worker learned with mini-batches of size 64, using the Adam optimizer and a learning rate of 1e−4.

        
# class StateRN(BasicModel):
#     def __init__(self, args):
#         super(StateRN, self).__init__(args, 'StateRN')

#         self.relation_type = args.relation_type


#         ##(number of filters per object+coordinate of object)*2+question vector
#         self.g_fc1 = nn.Linear((4*2)+11, 512)

#         #4 MLP layers 512 per layer
#         self.g_fc2 = nn.Linear(512, 512)
#         self.g_fc3 = nn.Linear(512, 512)
#         self.g_fc4 = nn.Linear(512, 512)

#         # A three-layer MLP consisting of
#         # 512, 1024 (with 2% dropout) and 29 units with ƒReLU non-linearities
#         # was used for fθ.
#         # self.f_fc1 = nn.Linear(512, 1024)
#         # self.f_fc2 = nn.Linear(512, 1024)
#         # self.f_fc3 = nn.Linear(512, 1024)
#         # self.f_fc4 = nn.Linear(64, 10)
#         self.f_fc1 = nn.Linear(512, 512)
#         self.f_fc2 = nn.Linear(512, 1024)
#         self.f_fc3 = nn.Linear(1024, 29)
#         # (?) #final output length depends on the answer embedding (10 in this case?)

#         self.coord_oi = torch.FloatTensor(args.batch_size, 2)
#         self.coord_oj = torch.FloatTensor(args.batch_size, 2)
#         if args.cuda:
#             self.coord_oi = self.coord_oi.cuda()
#             self.coord_oj = self.coord_oj.cuda()
#         self.coord_oi = Variable(self.coord_oi)
#         self.coord_oj = Variable(self.coord_oj)

#         # prepare coord tensor
#         def cvt_coord(i):
#             return [(i / 5 - 2) / 2., (i % 5 - 2) / 2.]

#         self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
#         if args.cuda:
#             self.coord_tensor = self.coord_tensor.cuda()
#         self.coord_tensor = Variable(self.coord_tensor)
#         np_coord_tensor = np.zeros((args.batch_size, 25, 2))
#         for i in range(25):
#             np_coord_tensor[:, i, :] = np.array(cvt_coord(i))
#         self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

#         self.fcout = FCOutputModel()

#         self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

#     def forward(self, img, qst):
#         # 4 columns,corresponds to object size (5x5 in conv)
#         # 6 objects, corresponds to img size = 256 pixels (?)
#         # 64 = minibatch size
#         # x = self.conv(img)  ## x = (64 x 24 x 5 x 5)
#         x = img # x = (64 x 6 x 4)
#         # x.numpy()
#         # x = x.T
#         # #obj should only have 3D
#         # print(len(x[0]))
#         # print('------')
#         # print(len(x[1]))
        
        
#         """g"""
#         mb = x.size()[0]
#         # mb = x.size()[0]
#         n_channels = x.size()[1]
#         # n_channels = x.size()[1]
#         print(n_channels)
#         d1 = x.size()[2]
#         # d2 = 1
#         # d = x.size()[0]
#         d = x.size()[2]
#         print(d)

#         # x_flat = x
#         # x_flat = x.view(mb, n_channels, d * d).permute(0, 2, 1) 
#         x_flat = x.view(mb, n_channels, d * d)
#         print("x_flat size: {}".format(len(x_flat[0])))
        
#         # print("coord_tensor {}".format(len(self.coord_tensor)))
#         # x_flat = torch.cat([x_flat, self.coord_tensor],2)
        

#         # x_flat = x
#         #4 dimensions?

#         # add coordinates
#         # coord = torch.unsqueeze(self.coord_tensor, 2)  #  64 x 25 x 1 x (256 + 2)
#         # coord = coord.repeat(1, 1, 25, 1)  #  64 x 25 x 25 x (256 + 2 + 11)
#         # x_flat = torch.cat([x_flat, coord], 2)


#         # add question everywhere
#         qst = torch.unsqueeze(qst, 1)  # (64,11) -> (64,1,11)
#         # print("QST1 {}".format(len(qst)))
#         # qst = qst.repeat(1, 3, 1)  # 64 x 4 x 11
#         qst = qst.repeat(1, 25, 1)  # 64 x 4 x 11
#         # print("QST2 {}".format(len(qst)))
#         qst = torch.unsqueeze(qst, 2)  # 64 x 4 x 1 x 11
#         print("QST {}".format(len(qst)))
        
        
        
#         x_i = torch.unsqueeze(x_flat, 1)  # 64 x 1 x 25 x (256 + 2)
#         # print(len(x_i))
#         x_i = x_i.repeat(1, 25, 1, 1)  # 64 x 25 x 25 x (256 + 2)
        
#         x_j = torch.unsqueeze(x_flat, 2)  #  64 x 25 x 1 x (256 + 2)
#         x_j = torch.cat([x_j, qst], 3) # 64 x 25 x 1 x (256 + 2 + 11)
#         x_j = x_j.repeat(1, 1, 25, 1)  #  64 x 25 x 25 x (256 + 2 + 11)

#         # cast all pairs against each other
#         # x_flat: 64 x 24 x 6
#         # x_i = torch.unsqueeze(x_flat, 1)  # 64 x 1 x 4 x 6
#         # print(x_i[0])
#         # x_i = x_i.repeat(1, 6, 1, 1)  # 64 x 4 x 4 x 6
#         # print("size of x_i{}".format(len(x_i[2])))
        
#         x_j = torch.unsqueeze(x_flat, 2)  # 64 x 4 x 1 x 6
#         # x_j = torch.cat([x_j,qst], 3)  # 64 x 4 x 1 x (6 + 11)
#         x_j = x_j.repeat(1, 1, 3, 1)  # 64 x 4 x 4 x (6 + 11)

#         # concatenate all together
#         print("size of x_i {}".format(len(x_i[2])))
#         print("size of x_j {}".format(len(x_j[2])))
#         x_full = torch.cat([x_i,x_j], 3)  # 64 x 4 x 4 x ((6) + (6 + 11))

#         # reshape for passing through network
#         x_ = x_full.view(mb * (d * d) * (d * d), 2 * (4+2) + 11)  # (64*4*4 x (2*6+11)) = (1024, 23)

#         x_ = self.g_fc1(x_)  # 64*4*4 x 512
#         x_ = F.relu(x_)
#         x_ = self.g_fc2(x_)  # 64*4*4 x 512
#         x_ = F.relu(x_)
#         x_ = self.g_fc3(x_)  # 64*4*4 x 512
#         x_ = F.relu(x_)
#         x_ = self.g_fc4(x_)  # 64*4*4 x 512
#         x_ = F.relu(x_)

#         # reshape again and sum
#         x_g = x_.view(mb, (d * d) * (d * d), 512)

#         x_g = x_g.sum(1).squeeze()  # 64 x 512

#         """f"""
#         # unsure of these dimensions
#         x_f = self.f_fc1(x_g)  # 64 x 1024
#         x_f = F.relu(x_f)
#         # x_f = self.f_fc2(x_f)  # 64 x 1024
#         # x_f = F.relu(x_f)
#         # x_f = F.dropout(x_f) # 2%
#         # # nn.Dropout(0.02)
#         # x_f = self.f_fc3(x_f)  # 64 x 1024
#         # x_f = F.log_softmax(x_f, dim=1)  # 64 x 10

#         return self.fcout(x_f)