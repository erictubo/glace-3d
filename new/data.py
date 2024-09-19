from pathlib import Path

path_to_scenes = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Data/GLACE')

scenes = {
        'pantheon_real': {
            'pantheon_real_N2_K1_M0': {
                'encoder_name': 'ace_encoder_pretrained.pt',
                'num_head_blocks': 2, # N
                'num_decoder_clusters': 1, # K
                'mode': 0, # M
                'sparse': True, # true for real data
            },
            'pantheon_real_N2_K1_M1': {
                'encoder_name': 'ace_encoder_pretrained.pt',
                'num_head_blocks': 2,
                'num_decoder_clusters': 1,
                'mode': 1,
                'sparse': True, # true for real data
            }
        },
        # 'pantheon_B': { },
        # 'pantheon_C': { },
        
        # 'notre_dame_real': { },
        # 'notre_dame_B': { },
        # 'notre_dame_E': { },

        # 'reichstag_real': { },
        # 'reichstag_A': { },
        # 'reichstag_B': { },

        # 'brandenburg_gate_real': { },
        # 'brandenburg_gate_B': { },
        # 'brandenburg_gate_C': { },
    }
    