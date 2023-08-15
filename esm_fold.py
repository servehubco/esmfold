import csv
import io
import logging
import zipfile
from pathlib import Path
from typing import Annotated, List, Optional

import esm
from esm.data import read_fasta
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from ray import serve
from pydantic import BaseModel, Field
import tempfile
import biotite.structure.io as bsio
from starlette.responses import StreamingResponse

from timeit import default_timer as timer

from esm_code.fold import create_batched_sequence_datasest

# Load ESM-1b model


app = FastAPI()


class SequenceInput(BaseModel):
    """
    A sequence input to the model.
    """
    name: Optional[str] = Field(None, description="Name for the sequence. If not provided, the sequence will be named the first 20 letters of the sequence.")
    sequence: str = Field(description="The protein sequence to fold.")


class FoldOutput(BaseModel):
    """
    A folded sequence output from the model.
    """
    name: str = Field(description="Name of the sequence.")
    sequence: str = Field(description="The protein sequence.")
    pdb_string: str = Field(description="The pdb string of the folded sequence. Save this string to a .pdf file.")
    mean_plddt: float = Field(description="The mean pLDDT of the folded sequence.")
    ptm: float = Field(description="The pTM of the folded sequence.")


class ModelInferenceHyperparameters(BaseModel):
    """
    Hyperparameters that can be set for model inference. These can be used to manage memory usage during inference.
    """
    chunk_size: Optional[int] = Field(None, description="Chunks axial attention computation to reduce memory usage from O(L^2) to O(L). Equivalent to running a for loop over chunks of each dimension. Lower values will result in lower memory usage at the cost of speed. Recommended values: 128, 64, 32. Default: None.")

# class FoldOutput(BaseModel):
#     pdb: str
#     plddt: Optional[pLDDTOutput]

@serve.deployment(route_prefix="/",
                  ray_actor_options={"num_cpus": 3, "num_gpus": 1})
@serve.ingress(app)
class MyFastAPIDeployment:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.log(logging.INFO, "Loading model...")
        self.model = esm.pretrained.esmfold_v1()
        self.logger.log(logging.INFO, "Model loaded.")
        self.model = self.model.eval().cuda()
        self.logger.log(logging.INFO, "Model set to eval and cuda.")


    @app.post("/set_model_inference_hyperparams")
    async def set_model_inference_hyperparams(self, params: ModelInferenceHyperparameters):
        """
        Set the model inference hyperparameters. This can be used to manage memory usage during inference.
        :return: A dictionary with the previous and current values of the chunk_size parameter.
        """
        self.model.set_chunk_size(params.chunk_size)
        return {
            "prev_params": {"chunk_size": self.model.trunk.chunk_size},
            "curr_params": {"chunk_size": params.chunk_size}
        }

    @app.post("/fold_sequences")
    async def fold_sequences(self, seqs: Annotated[List[SequenceInput], Body()] = Field(description="A list of sequences to fold with "
                                                                                 "names. "
                                                                                 "Use the `fold_sequences/no_name` "
                                                                                 "endpoint "
                                                                                 "if you don't have names."),
                             num_recycles: Annotated[int, Body()] = Field(4, description="Number of recycles to run. Defaults to number "
                                                                      "used in training (4)."),
                             max_tokens_per_batch: Annotated[int, Body()] = Field(1024,
                                                               description="Maximum number of tokens per gpu "
                                                                           "forward-pass. This will group shorter "
                                                                           "sequences together for batched prediction. "
                                                                           "Lowering this can help with out of memory "
                                                                           "issues, if these occur on short sequences. "
                                                                           "Default: 1024.")) -> List[FoldOutput]:
        """
        Fold a list of sequences.
        :return: A list of FoldOutput objects, each containing the name, sequence, pdb_string, mean_plddt, and ptm of the folded sequence.
        """
        for seq_input in seqs:
            if not seq_input.name:
                seq_input.name = seq_input.sequence[:20]

        # convert to a list of tuples
        seqs = [(seq_input.name, seq_input.sequence) for seq_input in seqs]

        batched_sequences = create_batched_sequence_datasest(seqs, max_tokens_per_batch)

        num_completed = 0
        num_sequences = len(seqs)
        outputs = []
        for headers, sequences in batched_sequences:
            start = timer()
            try:
                output = self.model.infer(sequences, num_recycles=num_recycles)
            except RuntimeError as e:
                if e.args[0].startswith("CUDA out of memory"):
                    if len(sequences) > 1:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed (CUDA out of memory) to predict batch of size {len(sequences)}. "
                            "Try lowering the max_tokens_per_batch parameter."
                        )
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed (CUDA out of memory) on sequence {headers[0]} of length {len(sequences[0])}. "
                                   f"Try lowering the max_tokens_per_batch parameter, or setting the chunk size with the"
                                   f"`set_model_inference_hyperparameters` endpoint."
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Exception {e} occurred while predicting sequence {headers[0]} of length {len(sequences[0])}: {e}."
                    )

            output = {key: value.cpu() for key, value in output.items()}
            pdbs = self.model.output_to_pdb(output)
            tottime = timer() - start
            time_string = f"{tottime / len(headers):0.1f}s"
            if len(sequences) > 1:
                time_string = time_string + f" (amortized, batch size {len(sequences)})"

            for header, seq, pdb_string, mean_plddt, ptm in zip(
                    headers, sequences, pdbs, output["mean_plddt"], output["ptm"]
            ):
                res = FoldOutput(
                    name=header,
                    sequence=seq,
                    pdb_string=pdb_string,
                    mean_plddt=mean_plddt,
                    ptm=ptm
                )
                outputs.append(res)
                num_completed += 1
                self.logger.info(
                    f"Predicted structure for {header} with length {len(seq)}, pLDDT {mean_plddt:0.1f}, "
                    f"pTM {ptm:0.3f} in {time_string}. "
                    f"{num_completed} / {num_sequences} completed."
                )

        return outputs

    @app.post("/fold_sequences/no_name")
    async def fold_sequences_no_name(self, seqs: Annotated[List[SequenceInput], Body()] = Field(description="A list of sequences to fold with "
                                                                                 "names. "
                                                                                 "Use the `fold_sequences/no_name` "
                                                                                 "endpoint "
                                                                                 "if you don't have names."),
                             num_recycles: Annotated[int, Body()] = Field(4, description="Number of recycles to run. Defaults to number "
                                                                      "used in training (4)."),
                             max_tokens_per_batch: Annotated[int, Body()] = Field(1024,
                                                               description="Maximum number of tokens per gpu "
                                                                           "forward-pass. This will group shorter "
                                                                           "sequences together for batched prediction. "
                                                                           "Lowering this can help with out of memory "
                                                                           "issues, if these occur on short sequences. "
                                                                           "Default: 1024.")) -> \
    List[FoldOutput]:
        """
        Fold a list of sequences.
        :return: A list of FoldOutput objects, each containing the name, sequence, pdb_string, mean_plddt, and ptm of the folded sequence.
        """
        seqs = [SequenceInput(sequence=seq) for seq in seqs]
        return await self.fold_sequences(seqs, num_recycles, max_tokens_per_batch)

    @app.post("/fold_sequence")
    async def fold_sequence(self, sequence: SequenceInput = Field(description="A sequence to fold with a name. Use the "
                                                                              "`fold_sequence/no_name` endpoint if "
                                                                              "you don't have a name."),
                             num_recycles: Annotated[int, Body()] = Field(4, description="Number of recycles to run. Defaults to number "
                                                                      "used in training (4).")) -> FoldOutput:
        """
        Fold a sequence.
        :return: A FoldOutput object containing the name, sequence, pdb_string, mean_plddt, and ptm of the folded sequence.
        """
        return (await self.fold_sequences([sequence], num_recycles))[0]


    @app.post("/fold_sequence/no_name")
    async def fold_sequence_no_name(self, sequence: SequenceInput = Field(description="A sequence to fold with a name. Use the "
                                                                              "`fold_sequence` endpoint if you'd like to provide a name for the sequence."),
                             num_recycles: Annotated[int, Body()] = Field(4, description="Number of recycles to run. Defaults to number "
                                                                      "used in training (4).")) -> FoldOutput:
        """
        Fold a sequence.
        :return: A FoldOutput object containing the name, sequence, pdb_string, mean_plddt, and ptm of the folded sequence.
        """
        return (await self.fold_sequences_no_name([sequence], num_recycles))[0]

    @app.post("/fold_fasta")
    async def fold_fasta(self, fasta: UploadFile = Field(description="A fasta file containing sequences to fold."), num_recycles: int = Field(4, description="Number of recycles to run. Defaults to number "
                                                                      "used in training (4)."),
                             max_tokens_per_batch: Annotated[int, Body()] = Field(1024,
                                                               description="Maximum number of tokens per gpu "
                                                                           "forward-pass. This will group shorter "
                                                                           "sequences together for batched prediction. "
                                                                           "Lowering this can help with out of memory "
                                                                           "issues, if these occur on short sequences. "
                                                                           "Default: 1024.")) -> List[FoldOutput]:
        """
        Fold sequences from a fasta file. Use the `fold_fasta/zipped` endpoint if you'd like to download the results as a zip file.
        :return: A list of FoldOutput objects, each containing the name, sequence, pdb_string, mean_plddt, and ptm of the folded sequence.
        """
        try:
            fasta_content = await fasta.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as temp_fasta:
                temp_fasta.write(fasta_content)
                temp_fasta_path = Path(temp_fasta.name)
                if temp_fasta_path.exists():
                    temp_fasta_path.unlink()

                all_sequences = sorted(read_fasta(temp_fasta), key=lambda header_seq: len(header_seq[1]))
                seqs = [SequenceInput(name=header, sequence=seq) for header, seq in all_sequences]
                return await self.fold_sequences(seqs, num_recycles, max_tokens_per_batch)

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


    @app.post("/fold_fasta/zipped")
    async def fold_fasta_zipped(self, fasta: UploadFile = Field(description="A fasta file containing sequences to fold."), num_recycles: int = Field(4, description="Number of recycles to run. Defaults to number "
                                                                      "used in training (4)."),
                             max_tokens_per_batch: Annotated[int, Body()] = Field(1024,
                                                               description="Maximum number of tokens per gpu "
                                                                           "forward-pass. This will group shorter "
                                                                           "sequences together for batched prediction. "
                                                                           "Lowering this can help with out of memory "
                                                                           "issues, if these occur on short sequences. "
                                                                           "Default: 1024.")):
        """
        Fold sequences from a fasta file and download the results as a zip file. Use the `fold_fasta` endpoint if you'd
        like to return the results as a list.
        :return: A zip file containing the pdb files and a csv file with the confidence metrics for each sequence.
        """
        try:
            results = await self.fold_fasta(fasta, num_recycles, max_tokens_per_batch)

            in_memory_zip = io.BytesIO()
            with zipfile.ZipFile(in_memory_zip, 'w') as zf:
                # Create the CSV in-memory
                csv_data = io.StringIO()
                csv_writer = csv.writer(csv_data)
                csv_writer.writerow(["Name", "Sequence", "mean_plddt", "ptm"])

                for result in results:
                    pdb_filename = f"{result.name}.pdb"
                    zf.writestr(pdb_filename, result.pdb_string)
                    csv_writer.writerow([result.name, result.sequence, result.mean_plddt, result.ptm])

                # Save the CSV to the zip
                csv_data.seek(0)
                zf.writestr("confidence_metrics.csv", csv_data.getvalue())

            in_memory_zip.seek(0)
            return StreamingResponse(in_memory_zip, media_type="application/zip",
                                     headers={"Content-Disposition": "attachment; filename=output.zip"})

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


deployment = MyFastAPIDeployment.bind()

