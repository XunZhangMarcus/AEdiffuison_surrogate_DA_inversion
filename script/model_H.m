function [yout] = model_H(ss,kfield,ii)
maindir = pwd;
addpath([maindir,'\Utilities']);
exampledir = [maindir,'\high_fidelity']; 

%%
t = (200:200:1000);
timestep = length(t);
load('\obscoor.mat');
nobs = size(obscoor,1);
y = zeros(timestep*nobs,1); 
head = zeros(nobs,1);
%%
xobs = obscoor(:,1)/1.5; 
yobs = obscoor(:,2)/1.5; 
%%
s_row = [34:1:93];
s_col = 12; 
s_lay = 1;
modifyssm(ii,exampledir,s_lay,s_row,s_col,ss);
%%
Tran=kfield;
dlmwrite([exampledir,'\parallel_',num2str(ii),'\Tran.dat'],Tran,'delimiter', '', 'precision', '%10.4f','newline', 'pc'); % 将新的K场写进dat文件中
%%
cd([exampledir,'\parallel_',num2str(ii)]);
system('mt3dms5b.bat');
cd(maindir);
%%
CC = readMT3D([exampledir,'\parallel_',num2str(ii),'\MT3D001.UCN']);
for i=1:size(CC,1) 
    Ctime(i)=CC(i).time;  
end
[m,n]=find(Ctime'==t);
for j = 1:timestep 
    tpcon=CC(m(j)).values;
    for k = 1:nobs    
        y(nobs*j-nobs+k,1) = tpcon(xobs(k),yobs(k)); 
    end    
end
%%
H = readDat([exampledir,'\parallel_',num2str(ii),'\zx_7_12.hed']);
tphead=H.values;
for k = 1:nobs    
    head(k,1) = tphead(xobs(k),yobs(k)); 
end
yout=[y;head];
end

