clc
clear all
close all
%%
g=5; % choose the benchmark function or comment and give as given below
%======================= Problem Initialization ===========================
fitnessfcn=@(x)func(x,g);
[nvar,LB,UB]=nvar_bound(g);
lincon=@(g)linear_constraint_gen(g);
nonlcon=@(x)nonlincon(x,g);

% fitnessfcn=@(x)norm(x);
% [nvar,LB,UB]=nvar_bound(g);
% [Aeq,beq,A,b]=linear_constraint_gen(g);
% nonlcon=@(x)nonlincon(x,g);
LB=LB';UB=UB';
%%
%======================= PSO - initialziation =============================
maxiter=300;
pop_size=200;
vmax=(UB-LB);
vmin=-vmax;

w=1.5;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=1.5;         % Personal Leopearning Coefficient
c2=2.0;         % Global Learning Coefficient


% % Constriction Coefficients
phi1=2.1;
phi2=2.1;
phi=phi1+phi2;
chi=2/(phi-2+sqrt(phi^2-4*phi));
w=chi;          % Inertia Weight
wdamp=1;        % Inertia Weight Damping Ratio
c1=chi*phi1;    % Personal Learning Coefficient
c2=chi*phi2;    % Global Learning Coefficient

pop_mat=zeros(nvar,pop_size);
velo=zeros(nvar,pop_size);

for i=1:pop_size
    pop_mat(:,i)=unifrnd(LB,UB,[nvar,1]);
    velo(:,i)=unifrnd(vmin,vmax,[nvar,1]);
end
costs=zeros(pop_size,maxiter);pbest=zeros(pop_size,maxiter);
pbest_pop=zeros(nvar,pop_size);gbest=zeros(maxiter,1);gbest_pop=zeros(nvar,maxiter);
for it=1:maxiter
    if it ~=1
        pop_mat=pop_mat+velo(:,:,it);
    end
    for j=1:pop_size
        pop_mat(:,j)=min(pop_mat(:,j),UB);
        pop_mat(:,j)=max(pop_mat(:,j),LB);
        costs(j,it)=func_penality(pop_mat(:,j),g,fitnessfcn,lincon,nonlcon,it/maxiter);
    end
    if it == 1
        pbest(:,1)=costs(:,1);
        pbest_pop=pop_mat;
        gbest(it)=min(costs(:,1));
        gbest_pop(:,it)=pop_mat(:,find(costs(:,1)==min(costs(:,1)),1,'last'));
    else
        it
        pbest(:,it)=pbest(:,it-1);
        pbest((pbest(:,it) > costs(:,it)),it)=costs((pbest(:,it) > costs(:,it)),it);
%         pbest_pop=pbest_pop(:,it-1);
        pbest_pop(:,(pbest(:,it-1) > costs(:,it)))=pop_mat(:,(pbest(:,it-1) > costs(:,it)));
        gbest(it)=func_penality(gbest_pop(:,it-1),g,fitnessfcn,lincon,nonlcon,it);
        if min(pbest(:,it)) < gbest(it)
            gbest(it)=min(pbest(:,it));
            gbest_pop(:,it)=pbest_pop(:,find(pbest(:,it)==min(pbest(:,it)),1,'last'));
        else
            gbest(it)=gbest(it-1);
            gbest_pop(:,it)=gbest_pop(:,it-1);
        end
        
       
    end
    velo(:,:,it+1)=w*velo(:,:,it)+(c1*rand(size(pop_mat)).*-(pop_mat-pbest_pop))...
        +(c2*rand(size(pop_mat)).*-(pop_mat-gbest_pop(:,it)));
    for j=1:pop_size
        velo(:,j,it+1)=min(velo(:,j,it+1),vmax);
        velo(:,j,it+1)=max(velo(:,j,it+1),vmin);
    end
    w=w*wdamp;
end
func(gbest_pop(:,maxiter),g)
