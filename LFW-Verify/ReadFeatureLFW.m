addpath '/home/wujiyang/caffe/matlab';

caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

% ROIx = 19:82;
% ROIy = 19:82;
ROIx = 1:96; %96 112
ROIy = 1:112;
feature_dim = 512; % 特征维度，记得修改
mean_value = 127.5;
scale = 0.0078125;

height = length(ROIx);
width = length(ROIy);

allPairs = [same_pair;diff_pair];
% meanC = caffe.read_mean('D:\ThirdPartyLibrary\caffe\examples\siamese\mean.proto');
net = caffe.Net('/home/wujiyang/FaceDataSetTools/FaceModels/amsoftmax-20/amsoftmax_20_deploy_mirror_normalize.prototxt','/home/wujiyang/FaceDataSetTools/FaceModels/amsoftmax-20/amsoftmax_20_vggface2.caffemodel', 'test');%
num = size(allPairs,1);
AllFeature1 = zeros(feature_dim,num);
AllFeature2 = zeros(feature_dim,num);
for i = 1 : floor(num/10)
    disp([i floor(num/10)]);
    J = zeros(height,width,3,10,'single');
    for j = 1 : 10
        I = imread(allPairs{(i-1)*10+j,1});
        I = permute(I,[2 1 3]);
        I = I(:,:,[3 2 1]);
        I = I(ROIx,ROIy,:);
        I = single(I) - mean_value;
        J(:,:,:,j) = I*scale;
%         J(:,:,1,j) = I(end:-1:1,:);
    end;
    f1 = net.forward({J});
    f1 = f1{1};
    AllFeature1(1:feature_dim,(i-1)*10+1:i*10) = reshape(f1,[size(AllFeature1,1),10]);
%     layer_conv52 = net.blob_vec(net.name2blob_index('pool5'));
%     conv52 = layer_conv52.get_data();
%     sum(conv52(:)>0) /320/100
end;
J = zeros(height,width,3,10,'single');
for j = 1 : num - floor(num/10) * 10
    I = imread(allPairs{floor(num/10) * 10+j,1});
    I = permute(I,[2 1 3]);
    I = I(:,:,[3 2 1]);
    I = I(ROIx,ROIy,:);
    I = single(I) - mean_value;
    J(:,:,:,j) = I*scale;
end;
f1 = net.forward({J});
f1=f1{1};
f1 = squeeze(f1);
AllFeature1(1:feature_dim,floor(num/10) * 10+1:num) = f1(:,1 : num - floor(num/10) * 10);

for i = 1 : floor(num/10)
    disp([i floor(num/10)]);
    J = zeros(height,width,3,10,'single');
    for j = 1 : 10
        I = imread(allPairs{(i-1)*10+j,2});
        I = permute(I,[2 1 3]);
        I = I(:,:,[3 2 1]);
        I = I(ROIx,ROIy,:);
        I = single(I) - mean_value;
        J(:,:,:,j) = I*scale;
    end;
    f1 = net.forward({J});
    f1 = f1{1};
    AllFeature2(1:feature_dim,(i-1)*10+1:i*10) = reshape(f1,[size(AllFeature2,1),10]);
end;
J = zeros(height,width,3,10,'single');
for j = 1 : num - floor(num/10) * 10
    I = imread(allPairs{floor(num/10) * 10+j,2});
    I = permute(I,[2 1 3]);
    I = I(:,:,[3 2 1]);
    I = I(ROIx,ROIy,:);
    I = single(I) - mean_value;
    J(:,:,:,j) = I*scale;
end;
f1 = net.forward({J});
f1=f1{1};
f1 = squeeze(f1);
AllFeature2(1:feature_dim,floor(num/10) * 10+1:num) = f1(:,1 : num - floor(num/10) * 10);
% caffe.reset_all();