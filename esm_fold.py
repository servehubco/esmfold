import logging

import torch
import esm
from fastapi import FastAPI
from ray import serve
from pydantic import BaseModel
import tempfile
import biotite.structure.io as bsio

# Load ESM-1b model


app = FastAPI()

class SequenceInput(BaseModel):
    sequence: str

@serve.deployment(route_prefix="/hello",
                  ray_actor_options={"num_cpus": 2, "num_gpus": 1})
@serve.ingress(app)
class MyFastAPIDeployment:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.logger.log(logging.INFO, "Loading model...")
        self.model = esm.pretrained.esmfold_v1()
        self.logger.log(logging.INFO, "Model loaded.")
        self.model = self.model.eval().cuda()
        self.logger.log(logging.INFO, "Model set to eval and cuda.")

    @app.post("/")
    async def root(self, seq_input: SequenceInput):
        try:
            with torch.no_grad():
                output = self.model.infer_pdb(seq_input.sequence)

            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=True) as temp:
                temp.write(output.encode())
                temp.flush()  # Ensure all data is written to the file

                struct = bsio.load_structure(temp.name, extra_fields=["b_factor"])
                pLDDT = struct.b_factor.mean()

            return {"pLDDT": pLDDT, "pdb": output}
        except Exception as e:
            raise e
            # raise HTTPException(status_code=400, detail=str(e))


deployment = MyFastAPIDeployment.bind()

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
# model.set_chunk_size(128)

# sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
# Multimer prediction can be done with chains separated by ':'

# with torch.no_grad():
#     output = model.infer_pdb(sequence)

# with open("result.pdb", "w") as f:
#     f.write(output)

# import biotite.structure.io as bsio
# struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
# print(struct.b_factor.mean())  # this will be the pLDDT
# 88.3