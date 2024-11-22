clear;clc;close all
%%
currentdir = pwd;
addpath([currentdir,'\Utilities']);
gcp;
%%
load Par.mat
Ne = Par.Ne;
cd([currentdir,'\high_fidelity'])
copyexample(Ne);
cd(currentdir);
%%
load Srange % range for contaminant source
ss_Ne = genex(Srange,Ne);
save('ss_200.mat',"ss_Ne")
%%
load('\z_vae_100.mat');
x1 = [z_vae_100;ss_Ne]; 
save('x1.mat',"x1") 

Nobs = 25*5+25;
conc_head_Ne = nan(Nobs,Ne);
load("\cond_200.mat") 
%%
tic
parfor i = 1:Ne
    [conc_head_Ne(:,i)] = model_H(ss_Ne(:,i),cond_200(:,i),i);
end
toc

save('y1.mat',"conc_head_Ne") 