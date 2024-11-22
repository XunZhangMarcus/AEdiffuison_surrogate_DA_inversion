function forward_model()
distcomp.feature( 'LocalUseMpiexec', false );  
Parallel_Computing = 1;
if Parallel_Computing == 1
    Ncpu = 6;
    myCluster = parcluster('local');
    myCluster.NumWorkers = Ncpu;
    saveProfile(myCluster);
    N = maxNumCompThreads;
    LASTN = maxNumCompThreads(N);
    LASTN = maxNumCompThreads('automatic');
    parpool('local',Ncpu);
end

load Par.mat
Ne = Par.Ne;
Nobs = Par.Nobs;
load ('your_path\z_a.mat')
load ('your_path\ss_a.mat')
load ('your_path\cond_a.mat')
a = ss_a;
b = cond_a;

ya = nan(Nobs,Ne);

parfor i = 1:Ne
    [ya(:,i)] = model_H(a(:,i),b(:,i),i);
end

save('ya.mat', "ya")

delete(gcp('nocreate'));
end