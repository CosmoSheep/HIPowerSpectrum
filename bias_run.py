import pandas as pd
import numpy as np
import bias_Mb_check
import matplotlib.pyplot as plt

for i in range(4):
	z_obs=[3.5,4.0,4.5,5.0,5.5]
	b1=[]
	b2=[]
	bs2=[]
	bv=[]
	bv_direct=[]
	bv_indirect=[]
	bv_indirect_sink=[]
	dxi_dz=[]

	for z in z_obs:
		xi= bias_Mb_check.b_sink(12.0,6.0,'fiducial')
		xi_sv, xi_nv = xi.xi_func()
		rho=bias_Mb_check.bias_v(12.0,z)

		rho_sv = rho.rho_HI('fmass_'+str(i+1)+'.csv',xi_sv)[0]
		rho_nv = rho.rho_HI('fmass_'+str(i+1)+'-nv.csv',xi_nv)[0]

		bv_tmp=np.log(rho_sv/rho_nv)
		bv_direct_tmp=rho.rho_HI_direct('fmass_'+str(i+1)+'.csv','fmass_'+str(i+1)+'-nv.csv',xi_sv,xi_nv)
		b1_tmp,bs2_tmp=rho.bias1_func()
		b2_tmp=rho.bias2_func()

		b1.append(b1_tmp)
		b2.append(b2_tmp)
		bv.append(bv_tmp)
		bs2.append(bs2_tmp)
		bv_direct.append(bv_direct_tmp)
		bv_indirect.append(bv_tmp-bv_direct_tmp)
		bv_indirect_sink.append(xi.b())
		dxi_dz.append(np.sum(xi.dxi_dz()))
		# print(xi.dxi_dz())
		
		print('For fiducial reionization')
		print('At z_obs='+str(z)+', bv='+str(bv_tmp)+',bv_direct='+ str(bv_direct_tmp)+',bv_ind='+str(bv_tmp-bv_direct_tmp))
		print('b1='+str(b1_tmp))
		print('bs2='+str(bs2_tmp))
		print('b2='+str(b2_tmp))

	tb={'z_obs':z_obs,'b1':b1,'b2':b2,'bs2':bs2, 'bv':bv, 'bv_dir':bv_direct, 'bv_ind':bv_indirect,'bv_ind_sink':bv_indirect_sink,'dxi_dz':
	dxi_dz}
	pd.DataFrame(tb).to_csv('bias_table_'+str(i+1)+'_v3.csv')
