            if LOCAL_STRUCTURE=='trigonal_pyramid':
                sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_pyramid_monodentate(domain=vars()['domain'+str(int(i+1))+'A'],top_angle=70,phi=0,r=2,attach_atm_ids=ids,offset=offset,pb_id=SORBATE_id,O_id=O_id,mirror=MIRROR[i])           
            elif LOCAL_STRUCTURE=='octahedral':
                sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_octahedral_monodentate(domain=vars()['domain'+str(int(i+1))+'A'],phi=0,r=2,attach_atm_id=ids,offset=offset,sb_id=SORBATE_id,O_id=O_id)           
            elif LOCAL_STRUCTURE=='tetrahedral':
                pass#to be completed
            elif LOCAL_STRUCTURE=='arbitrary_polyhedral':
                pass#to be completed