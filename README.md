# comparative-dialogue
A repository for experiments with building a (neural) comparative dialogue system

# simple pipeline 
zero-version of pipeline

The pipeline consists of 3 parts:

1) Recognition of objects for compare and aspects from user's natural language's question. This functionality provided by external pretrained model. The best NER model(at this moment) is situated here https://drive.google.com/file/d/1ka0VtcHDqy0gEOpn8bPusOEVdh2FL4RU/view?usp=sharing.
You should put it to 
/home/USER/comparative-dialogue/comparative-dialogue/external_pretrained_models/ or similar way
The NER may not work out properly( does not extract 2 object) which due to error.
In this case in \*.ipynb you can set list of aspects by hands. (fill the *objects_list* with the corresponding pair of words)

2) create query by 2 extracted objects names and aspects if it exists, recieve *.json file

3) preprocess the recieved *.json file and generate answer.
In generation/generation.py there is simple class for generation, which may be complicated in the future.

How to run:

in <>.ipynb 

or from command line, i.e. *python3 simple_pipeline.py --input "['what is better amazon or itunes for showing', 'what is better mouse or rat', 'what is easier to make bread o pizza']"*
assertion means at least one from 2 objects for comparation is not recognized by NER


