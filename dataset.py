import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

class MyDataset(Dataset):
    def __init__(self, csv_path,max_len = None,is_twist = False):
        self.data = pd.read_csv(csv_path)
        self.one_hot_map = pd.read_pickle('../one_hot_map.pkl')
        if max_len is not None:
            self.data = self.data[self.data.groupby('纱批')['纱批'].transform('count') <= max_len]
        self.unique_name = self.data['纱批'].unique()
        self.max_patch_count = max_len if max_len is not None else max([len(self.data[self.data['纱批'] == name]) for name in self.unique_name])
        
        # HVI指标
        self.feature_cols = ['棉等级','MIC','MAT%','LEN(INCH)','SFI(%)','STR(CN/TEX)']
        # 物料名称使用比例
        self.p = '物料名称使用比例'
        # ps 纺纱纱支
        self.yarn_count = '纺纱纱支'
        # sp 纺纱方式
        self.yarn_method = '纺纱方式'
        
        # 股线控制参数
        self.is_twist = is_twist
        self.single_twist = ['纺纱股数','单纱捻度']
        self.ply_twist = ['纺纱股数','股线捻度']
        
        # 梳棉,不考虑工艺路线
        self.comber = ['梳棉工艺名','精梳工艺名']
        # self.comber = ['工艺路线','梳棉工艺名','精梳工艺名']
        
        self.counts = len([name for name in self.unique_name if len(self.data[self.data['纱批'] == name]) < max_len]) if max_len is not None else len(self.unique_name)

        # self.data['棉等级'] = self.data['棉等级'] / 100.
        self.data['物料名称使用比例'] = self.data['物料名称使用比例'] / 100.
        self.data['纱强力'] = self.data['纱强力'] / 100.
                
        # # 对全局数据进行归一化
        # self.scaler = MinMaxScaler()
        # # MAT%存在0,是否归一化待定
        # self.data[['MIC','LEN(INCH)','SFI(%)','STR(CN/TEX)']] = self.scaler.fit_transform(self.data[['MIC','LEN(INCH)','SFI(%)','STR(CN/TEX)']])
        # self.data[['纺纱纱支','捻度']] = self.scaler.fit_transform(self.data[['纺纱纱支','捻度']])
        
    def get_onehot_by_name(self,name):
        return self.one_hot_map[str(name)]
        
    def get_max_patch_count(self):
        return self.max_patch_count

    def __len__(self):
        return self.counts

    # 对每组纱批数据进行归一化,可选(不适合用min_max，如果只有两个数据会归一化成0和1，丧失相对比例关系)
    def _select_feature_cols(self, data_group):
        # HVI指标（基础）与 AFIS 指标
        self.base_hvi_cols = ['MIC','MAT%','LEN(INCH)','SFI(%)','STR(CN/TEX)']
        self.afis_cols = ['afis-细度','afis-成熟度','afis-均长','afis-短绒率','afis-强度']
        # 若当前纱批的 HVI 合计为 0，则切换为 AFIS 特征
        use_afis = data_group[self.base_hvi_cols].sum().sum() == 0
        feature_cols = ['棉等级'] + (self.afis_cols if use_afis else self.base_hvi_cols)
        return feature_cols

    def __norm(self, data, feature_cols):
        data[feature_cols] = data[feature_cols].apply(
            lambda col: col / col.sum() if col.sum() != 0 else 0.0
        )
        return data

    def __getitem__(self, idx):
        name = self.unique_name[idx]
        tmp_data = self.data[self.data['纱批'] == name]
        length = len(tmp_data)

        # HVI/AFIS 参数进行归一化
        hvi_cols = self.base_hvi_cols
        afis_cols = self.afis_cols
        tmp_data = self.__norm(tmp_data, hvi_cols)
        tmp_data = self.__norm(tmp_data, afis_cols)

        # HVI参数进行归一化
        feature_cols = self._select_feature_cols(tmp_data)

        # 将棉等级对应的 one-hot 编码拼接到特征中
        degreed = np.array([self.get_onehot_by_name(n) for n in tmp_data['棉等级'].values],dtype=np.float32)
        feature1 = torch.from_numpy(np.concatenate([degreed, tmp_data[feature_cols[1:]].values], axis=1)).float()
        
        # 将梳棉特征转为one-hot编码
        comber_onehots = [np.array([self.get_onehot_by_name(x) for x in tmp_data[col].values], dtype=np.float32) for col in self.comber]
        # 按列拼接
        comber_feature = np.concatenate(comber_onehots, axis=1)
        comber_feature = torch.from_numpy(comber_feature).float()

        # 物料使用比例
        p = torch.from_numpy(tmp_data[self.p].values).float()
        
        # ps
        # 对于单纱将单纱股数和捻度作为ps输入
        ps = [tmp_data[self.yarn_count].values[0]] if self.is_twist else tmp_data[[self.yarn_count] + self.single_twist].values[0] 
        
        # 对于多股纱，这里不加入股线参数
        # ps = [tmp_data[self.yarn_count].values[0]]
        
        # sp
        sp = self.get_onehot_by_name(tmp_data[self.yarn_method].values[0])
        sp_ps = torch.from_numpy(np.array(np.concat((sp,ps),axis=0),dtype=np.float32)).float()

        # 股线控制参数
        ## 单纱参数 self.single_twist
        ## 股线参数 self.ply_twist
        twist = torch.from_numpy(tmp_data[self.ply_twist].values[0]).float()
        
        # 标签 实际强力值
        labels = torch.tensor(tmp_data['纱强力'].values[0]).float()

        # 对输入特征进行填充，统一到同一维度
        pad1 =  nn.ZeroPad2d(padding=(0,0,0,(self.max_patch_count - length)))
        feature1 = pad1(feature1)
        comber_feature = pad1(comber_feature)
        
        pad2 = nn.ZeroPad2d(padding=(0,(self.max_patch_count - length)))
        p = pad2(p)
        
        # B batch_size 
        # N self.max_patch_count
        # HVI特征,物料使用比例,梳棉特征,sp_ps,股线控制参数,标签
        # (B,N,9),(B,N),(B,N,21(7+14)),(B,3(多股)|5(单纱)),(B,2), (B,)
        # return feature1,p,comber_feature,sp_ps,twist,labels
        if self.is_twist:
            return feature1,p,comber_feature,sp_ps,twist,labels
        return feature1,p,comber_feature,sp_ps,labels
    
if __name__ == "__main__":
    dataset = MyDataset('../test_data.csv',max_len=3)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for i,(feature1,p,comber_feature,sp_ps,labels) in enumerate(dataloader):
        print(feature1.shape,p.shape,comber_feature.shape,sp_ps.shape,labels.shape)