def renameChannels(raw_combined):
    #changing ch_names from 1010 standard  "C1." -> "C1" 
    mapping={ch:ch.replace(".","") for ch in raw_combined.ch_names }
    raw_combined.rename_channels(mapping)
    
    return raw_combined

def bandFilter(raw_combined):
    raw_filter=raw_combined.filter(8.,30.,fir_design='firwin', skip_by_annotation='edge')
    
    return raw_filter