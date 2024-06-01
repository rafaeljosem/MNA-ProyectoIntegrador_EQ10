import os

from deepeval.synthesizer import Synthesizer

synth = Synthesizer()

list_docs = []

for path, folders, files in os.walk('env_data'):
    for file in files:
        if file.endswith('.txt'):
            list_docs.append(f'env_data/{file}')

synth.generate_goldens_from_docs(
    document_paths=list_docs,
    include_expected_output=True,
    chunk_overlap=50,
    max_goldens_per_document=2)

synth.save_as(file_type='json', directory='./synthetic_data')
