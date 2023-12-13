import transformers
import os
import time

SAVE_DIR = os.path.dirname(os.getcwd())

def main():
    '''
        Downloading the pre-trained models from transformers in advance to avoid network connection problem.
    '''

    prefix = 'EleutherAI/'
    model_list = ['pythia-70m-deduped','pythia-160m-deduped','pythia-410m-deduped','pythia-1b-deduped','pythia-1.4b-deduped']

    for model in model_list:
        try_cnt = 0
        while try_cnt<100:
            try:
                m=transformers.AutoModel.from_pretrained(prefix+model)
                t=transformers.AutoTokenizer.from_pretrained(prefix+model)
                break
            except Exception as e:
                print(str(e))
                print("Retry downloading %s"%(prefix+model))
                try_cnt+=1
                time.sleep(2)      

    revision_list = []  # e.g., ['step16','step10000']
    for model in model_list:
        for revision in revision_list:
            try_cnt = 0
            while try_cnt<100:
                try:
                    m=transformers.AutoModel.from_pretrained(prefix+model,
                                                        revision=revision,
                                                        cache_dir=os.path.join(os.path.join(SAVE_DIR,model),revision))
                    t=transformers.AutoTokenizer.from_pretrained(prefix+model,
                                                        revision=revision,
                                                        cache_dir=os.path.join(os.path.join(SAVE_DIR,model),revision))
                    break
                except Exception as e:
                    print(str(e))
                    print("Retry downloading %s"%(prefix+model))
                    try_cnt+=1
                    time.sleep(2)   

if __name__ == "__main__":
    main()