from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchdiffeq import odeint

app = FastAPI(title="ViroFlux API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # This wildcard allows any website to connect
    allow_credentials=True,
    allow_methods=["*"], # Allows all standard HTTP methods like GET and POST
    allow_headers=["*"], # Allows all headers
)
# Enable CORS so the GitHub Pages frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Model Definitions (Matching your trained architecture)
class SpikeVAE(nn.Module):
    def __init__(self, seq_len=1273, vocab_size=23, latent_dim=16, embed_dim=8):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        flat_dim = seq_len * embed_dim
        
        self.encoder_net = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len * vocab_size)
        )
        
    def encode(self, x):
        embedded = self.embedding(x)
        flat = embedded.view(embedded.size(0), -1)
        h = self.encoder_net(flat)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        flat_logits = self.decoder_net(z)
        return flat_logits.view(-1, self.seq_len, self.vocab_size)

class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim)
        )
    def forward(self, t, z):
        return self.net(z)

# 2. Initialization & Loading onto CPU
device = torch.device("cpu")
vae = SpikeVAE().to(device)
ode_func = LatentODEFunc().to(device)

try:
    vae.load_state_dict(torch.load("viroflux_vae.pt", map_location=device))
    ode_func.load_state_dict(torch.load("viroflux_ode.pt", map_location=device))
    vae.eval()
    ode_func.eval()
    models_loaded = True
except Exception as e:
    models_loaded = False
    print(f"Error loading models: {e}")

AMINO_ACIDS = ["-", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X", "*"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# 3. API Endpoints
class PredictionRequest(BaseModel):
    sequence: str
    t_start: float = 0.0
    t_end: float = 1.0

@app.get("/")
def read_root():
    return {"status": "ViroFlux API is online and ready!", "models_loaded": models_loaded}

@app.post("/predict")
def predict_trajectory(req: PredictionRequest):
    if not models_loaded:
        return {"error": "Models failed to load on server."}
    
    # Safely convert incoming amino acid string to index tensor
    try:
        seq_indices = [AA_TO_IDX.get(aa, 21) for aa in req.sequence] # 21 is 'X' (unknown)
        seq_tensor = torch.tensor([seq_indices], dtype=torch.long).to(device)
    except Exception as e:
        return {"error": f"Invalid sequence input. {e}"}

    with torch.no_grad():
        mu, _ = vae.encode(seq_tensor)
        
        # Continuous Neural ODE Integration
        if req.t_end > req.t_start:
            t_steps = torch.tensor([req.t_start, req.t_end], dtype=torch.float32).to(device)
            trajectory = odeint(ode_func, mu.squeeze(0), t_steps, method='dopri5')
            projected_latent = trajectory[-1:]
        else:
            projected_latent = mu
            
        pred_logits = vae.decode(projected_latent)
        pred_indices = torch.argmax(pred_logits, dim=-1).squeeze(0).tolist()
        pred_seq_chars = [AMINO_ACIDS[idx] for idx in pred_indices]
        
    pred_sequence_str = "".join(pred_seq_chars)
    
    # Detect Mutations
    mutations = []
    for i in range(len(req.sequence)):
        if req.sequence[i] != pred_sequence_str[i]:
            mutations.append(f"{req.sequence[i]}{i+1}{pred_sequence_str[i]}")
            
    return {
        "predicted_sequence": pred_sequence_str,
        "mutations": mutations,
        "mutation_count": len(mutations)
    }
