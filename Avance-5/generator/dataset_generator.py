# dataset_generator.py

"""
Module: legal_dataset_generator.py
Description: Module for generating legal datasets from Mexican legal documents.
"""

import json
import random
import time
from typing import Dict, List

from langchain.chains.llm import LLMChain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import TokenTextSplitter
from loader.legal_document_loader import LegalDocumentLoader
from openai import RateLimitError
from pydantic import BaseModel, Field


class QuestionAnsweringInstruction(BaseModel):
    """
    Instruction Model for the Question-Answering instruction type
    """
    instruction: str = Field(
        description="The instruction or question for the \
        Question Answering task.")
    output: str = Field(
        description="The corresponding answer or output for the \
            Question Answering instruction.")
    context: str = Field(
        description="The relevant excerpt from the legal document used to \
            generate the Question Answering instruction and output.")


class SummarizationInstruction(BaseModel):
    """
    Instruction Model for the Summarization instruction type
    """
    instruction: str = Field(
        description="The instruction or prompt for the Summarization task.")
    output: str = Field(
        description="The corresponding summary or output for \
            the Summarization instruction.")
    context: str = Field(
        description="The relevant excerpt from the legal document \
            used to generate the Summarization instruction and output.")


class LegalAdviceInstruction(BaseModel):
    """
    Instruction Model for the Legal Advice instruction type
    """
    instruction: str = Field(
        description="The instruction or prompt for the Legal Advice task.")
    output: str = Field(
        description="The corresponding advice or output for the \
            Legal Advice instruction.")
    context: str = Field(
        description="The relevant excerpt from the legal document used to \
            generate the Legal Advice instruction and output.")


class DocumentDraftingInstruction(BaseModel):
    instruction: str = Field(
        description="The instruction or prompt for the \
            Document Drafting task.")
    output: str = Field(
        description="The corresponding drafted document or output for the \
            Document Drafting instruction.")
    context: str = Field(
        description="The relevant excerpt from the legal document used to \
            generate the Document Drafting instruction and output.")


class Task(BaseModel):
    question_answering: List[QuestionAnsweringInstruction] = Field(
        description="A list of instruction-output pairs for the \
            Question Answering task.")
    summarization: List[SummarizationInstruction] = Field(
        description="A list of instruction-output pairs for \
            the Summarization task.")
    legal_advice: List[LegalAdviceInstruction] = Field(
        description="A list of instruction-output pairs for \
            the Legal Advice task.")
    document_drafting: List[DocumentDraftingInstruction] = Field(
        description="A list of instruction-output pairs for the \
            Document Drafting task.")


class Dataset(BaseModel):
    items: List[Task]


class DatasetGenerator:
    """
    Class for generating legal datasets from Mexican legal documents.
    """

    def __init__(self, llm: BaseLLM):
        """
        Initialize the DatasetGenerator.

        :param llm: The language model to use for generating legal datasets.
        """
        self._llm = llm

    def generate_from_source(
        self,
        source_type: str,
        source: str,
    ) -> List[Dict]:
        """
        Generate JSON objects for each article from
        the provided legal documents source.

        :param source_type: The type of source (url, csv, or dataframe).
        :param source: The source of legal documents
        (URL, file path, or DataFrame).

        :return: List of JSON objects representing legal documents.
        """
        if source_type == "url":
            legal_documents = LegalDocumentLoader.load_from_url(source)
        elif source_type == "csv":
            legal_documents = LegalDocumentLoader.load_from_csv(source)
        elif source_type == "parquet":
            legal_documents = LegalDocumentLoader.load_from_parquet(source)
        # elif source_type == "dataframe":
        #     legal_documents = LegalDocumentLoader.load_from_dataframe(source)
        else:
            raise ValueError(f"Invalid source type: {source_type}")

        return legal_documents

    def generate_from_legal_documents(
        self,
        legal_documents: List[Dict],
        # downstream_tasks: List[str],
        max_items_per_document: int = 2,
        # max_pairs_per_article: int = 2,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        document_limit: int = None
    ) -> Dataset:
        """
        Generate JSON objects for each article from
        the provided list of legal documents.

        :param legal_documents: List of legal documents as strings.
        :param max_items_per_document: Maximum number of items
        to extract from each document.

        :param chunk_size: The target size of each text chunk (default: 1000).
        :param chunk_overlap: The overlap size between
        adjacent chunks (default: 200).

        :return: List of JSON objects representing articles.
        """
        pairs = []

        for document in legal_documents[:document_limit]:

            # title = document["Title"]
            text = document["Text"]

            text_splitter = TokenTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            text_chunks = text_splitter.split_text(text)
            num_chunks = len(text_chunks)

            # Randomly select chunks based on max_items_per_document
            selected_chunk_indices = random.sample(
                range(num_chunks), min(max_items_per_document, num_chunks))
            selected_chunks = [text_chunks[i] for i in selected_chunk_indices]

            for chunk in selected_chunks:
                try:
                    # tasks = ", ".join(downstream_tasks)
                    # print(f"---\n\nThis are the tasks: {tasks}\n")
                    parser = JsonOutputParser(pydantic_object=Task)

                    template = template = """
                        Tomando en cuenta el fragmento de texto legal compartido, por favor genera dos ejemplos de pares instruction-output, 
                        para los siguientes tipos de tareas: "Question Answering (QA)", "Summarization", "Legal Advice Generation" y "Legal Document Drafting".

                        Para cada tarea, sigue las siguientes instrucciones:

                        Question Answering (QA):
                        - La instrucción debe ser una pregunta clara y específica basada en el contexto proporcionado, debe hacer referencia a los elementos clave del contexto, como número de artículo, nombre de la ley o similar.
                        - La salida debe ser una respuesta directa y concisa a la pregunta, utilizando la información del contexto.
                        - El contexto siempre debe comenzar al inicio de un artículo o capítulo, debe contener el texto de manera precisa y completa del artículo; puede contener más de un artículo, pero siempre contener el texto completo del o de los artículos seleccionados. Si el contenido del artículo contiene viñetas o numeración, asegúrate de incluirlo también exactamente como se presenta en el documento. Solamente no debes incluir anotaciones de tipo "Párrafo adicionado DOF 15-08-2008" o "Artículo reformado DOF 31-01-1974".

                        Summarization:
                        - La instrucción debe solicitar un resumen del contenido del artículo o sección proporcionada, debe hacer referencia a los elementos clave del contexto, como número de artículo, nombre de la ley o similar.
                        - La salida debe ser un resumen conciso que capture los puntos clave del contexto.
                        - El contexto siempre debe comenzar al inicio de un artículo o capítulo, debe contener el texto de manera precisa y completa del artículo; puede contener más de un artículo, pero siempre contener el texto completo del o de los artículos seleccionados. Si el contenido del artículo contiene viñetas o numeración, asegúrate de incluirlo también exactamente como se presenta en el documento. Solamente no debes incluir anotaciones de tipo "Párrafo adicionado DOF 15-08-2008" o "Artículo reformado DOF 31-01-1974".

                        Legal Advice Generation:
                        - La instrucción debe solicitar un consejo legal basado en el contexto proporcionado, debe hacer referencia a los elementos clave del contexto, como número de artículo, nombre de la ley o similar.
                        - La salida debe ser un consejo legal claro y relevante, considerando la información del contexto.
                        - El contexto siempre debe comenzar al inicio de un artículo o capítulo, debe contener el texto de manera precisa y completa del artículo; puede contener más de un artículo, pero siempre contener el texto completo del o de los artículos seleccionados. Si el contenido del artículo contiene viñetas o numeración, asegúrate de incluirlo también exactamente como se presenta en el documento. Solamente no debes incluir anotaciones de tipo "Párrafo adicionado DOF 15-08-2008" o "Artículo reformado DOF 31-01-1974".

                        Legal Document Drafting:
                        - La instrucción debe solicitar la redacción de un documento legal basado en el contenido del artículo o sección, debe hacer referencia a los elementos clave del contexto, como número de artículo, nombre de la ley o similar.
                        - La salida debe seguir la estructura: "CLAUSULA [número de cláusula en número ordinal o número romano, en mayúsculas].- [nombre de la cláusula]. [contenido de la cláusula]."
                        - El contexto siempre debe comenzar al inicio de un artículo o capítulo, debe contener el texto de manera precisa y completa del artículo; puede contener más de un artículo, pero siempre contener el texto completo del o de los artículos seleccionados. Si el contenido del artículo contiene viñetas o numeración, asegúrate de incluirlo también exactamente como se presenta en el documento. Solamente no debes incluir anotaciones de tipo "Párrafo adicionado DOF 15-08-2008" o "Artículo reformado DOF 31-01-1974".

                        El formato que deben seguir los ejemplos de tareas es el siguiente:\n\n{format_instructions}\n\n
                        
                        Contexto: {chunk}
                        """
                    # with open('generator/prompt.txt', 'r', encoding='utf8') as f:
                    #     template = "".join(f.readlines())

                    prompt = PromptTemplate(
                        template=template,
                        input_variables=["chunk"],
                        partial_variables={
                            "format_instructions":
                            parser.get_format_instructions(),
                        },
                    )

                    query = (
                        "Genera los ejemplos de pares instruction-output, "
                        f"con el siguiente fragmento de texto legal:\n{chunk}")

                    chain = LLMChain(llm=self._llm, prompt=prompt)
                    result = chain.invoke(query)

                    generated_doc_json = result['text']

                    generated_task_dict = json.loads(generated_doc_json)

                    generated_task = Task(**generated_task_dict)
                    pairs.append(generated_task)

                except RateLimitError:
                    print("You've reached your rate limit.",
                          " Finalizing generation process")
                    return
                except Exception as e:
                    print(f"Failed to generate JSON objects for chunk: {e}")
                    continue

                time.sleep(1)

        print(f"---\n\nThis are all the pairs: {pairs}")

        return Dataset(items=pairs)
