from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import random

app = FastAPI(title="BB84 Quantum Key Distribution API")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://bb84-front-end.vercel.app",
        "https://bb84-front-end.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BB84Request(BaseModel):
    num_bits: int = Field(..., ge=10, le=500, description="Number of bits to generate")
    eve_enabled: bool = Field(..., description="Enable eavesdropper")
    eve_interception_rate: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability Eve intercepts each qubit")


class BB84Response(BaseModel):
    alice_bits: List[int]
    alice_bases: List[str]
    encoded_qubits: List[str]
    eve_actions: Optional[dict]
    bob_bases: List[str]
    bob_results: List[int]
    matching_indices: List[int]
    sifted_key_alice: List[int]
    sifted_key_bob: List[int]
    qber: float
    errors: int
    sample_size: int
    eve_detected: bool
    status: str
    eve_interception_count: int


class QuantumChannel:
    """Simulates quantum channel with optional eavesdropping"""
    
    @staticmethod
    def encode_qubit(bit: int, basis: str) -> str:
        """Encode a classical bit into a quantum state"""
        if basis == '+':
            return '|0⟩' if bit == 0 else '|1⟩'
        else:  # basis == '×'
            return '|+⟩' if bit == 0 else '|−⟩'
    
    @staticmethod
    def measure_qubit(qubit_state: str, original_basis: str, measure_basis: str, original_bit: int) -> int:
        """
        Simulate quantum measurement
        Returns the measurement result based on basis matching
        """
        if measure_basis == original_basis:
            # Bases match - measurement returns original bit
            return original_bit
        else:
            # Bases don't match - random result (50/50)
            return random.randint(0, 1)


class BB84Protocol:
    """Complete BB84 Protocol Implementation"""
    
    def __init__(self, num_bits: int, eve_enabled: bool = False, eve_interception_rate: float = 0.5):
        self.num_bits = num_bits
        self.eve_enabled = eve_enabled
        self.eve_interception_rate = eve_interception_rate
        self.channel = QuantumChannel()
        
    def step1_alice_preparation(self):
        """Step 1: Alice generates random bits and bases"""
        alice_bits = [random.randint(0, 1) for _ in range(self.num_bits)]
        alice_bases = [random.choice(['+', '×']) for _ in range(self.num_bits)]
        return alice_bits, alice_bases
    
    def step2_alice_encoding(self, alice_bits: List[int], alice_bases: List[str]):
        """Step 2: Alice encodes bits into qubits"""
        encoded_qubits = [
            self.channel.encode_qubit(bit, basis)
            for bit, basis in zip(alice_bits, alice_bases)
        ]
        return encoded_qubits
    
    def step3_eve_interception(self, alice_bits: List[int], alice_bases: List[str]):
        """
        Step 3: Eve randomly intercepts qubits
        Eve intercepts with probability eve_interception_rate
        """
        eve_bases = []
        eve_measurements = []
        eve_intercepted = []
        eve_interception_count = 0
        
        for i in range(self.num_bits):
            # Randomly decide if Eve intercepts this qubit
            intercepts = random.random() < self.eve_interception_rate
            eve_intercepted.append(intercepts)
            
            if intercepts:
                eve_interception_count += 1
                # Eve chooses a random basis
                eve_basis = random.choice(['+', '×'])
                eve_bases.append(eve_basis)
                
                # Eve measures the qubit
                measured_bit = self.channel.measure_qubit(
                    None,  # Not needed for our simulation
                    alice_bases[i],
                    eve_basis,
                    alice_bits[i]
                )
                eve_measurements.append(measured_bit)
            else:
                # Eve doesn't intercept - qubit passes through unchanged
                eve_bases.append(None)
                eve_measurements.append(None)
        
        return {
            'bases': eve_bases,
            'measurements': eve_measurements,
            'intercepted': eve_intercepted,
            'interception_count': eve_interception_count
        }
    
    def step4_bob_measurement(self, alice_bits: List[int], alice_bases: List[str], eve_actions: Optional[dict]):
        """Step 4: Bob measures qubits with random bases"""
        bob_bases = [random.choice(['+', '×']) for _ in range(self.num_bits)]
        bob_results = []
        
        for i in range(self.num_bits):
            if self.eve_enabled and eve_actions['intercepted'][i]:
                # Eve intercepted - Bob measures Eve's re-encoded qubit
                eve_basis = eve_actions['bases'][i]
                eve_bit = eve_actions['measurements'][i]
                
                result = self.channel.measure_qubit(
                    None,
                    eve_basis,
                    bob_bases[i],
                    eve_bit
                )
            else:
                # No Eve or Eve didn't intercept - Bob measures Alice's qubit directly
                result = self.channel.measure_qubit(
                    None,
                    alice_bases[i],
                    bob_bases[i],
                    alice_bits[i]
                )
            
            bob_results.append(result)
        
        return bob_bases, bob_results
    
    def step5_basis_reconciliation(self, alice_bits: List[int], alice_bases: List[str], 
                                   bob_bases: List[str], bob_results: List[int]):
        """Step 5: Basis reconciliation - keep only matching bases"""
        matching_indices = [
            i for i in range(self.num_bits)
            if alice_bases[i] == bob_bases[i]
        ]
        
        sifted_key_alice = [alice_bits[i] for i in matching_indices]
        sifted_key_bob = [bob_results[i] for i in matching_indices]
        
        return matching_indices, sifted_key_alice, sifted_key_bob
    
    def step6_qber_calculation(self, alice_bits: List[int], bob_results: List[int], 
                               matching_indices: List[int]):
        """Step 6: Calculate QBER and detect eavesdropping"""
        # Use a portion of the sifted key for error checking
        sample_size = max(1, len(matching_indices) // 2)
        sample_indices = matching_indices[:sample_size]
        
        errors = sum(
            1 for i in sample_indices
            if alice_bits[i] != bob_results[i]
        )
        
        qber = (errors / sample_size * 100) if sample_size > 0 else 0
        
        # QBER threshold for Eve detection
        threshold = 11.0
        eve_detected = qber > threshold
        status = "Insecure Channel" if eve_detected else "Secure Channel"
        
        return qber, errors, sample_size, eve_detected, status
    
    def run_protocol(self):
        """Execute complete BB84 protocol"""
        # Step 1: Alice preparation
        alice_bits, alice_bases = self.step1_alice_preparation()
        
        # Step 2: Alice encoding
        encoded_qubits = self.step2_alice_encoding(alice_bits, alice_bases)
        
        # Step 3: Eve interception (if enabled)
        eve_actions = None
        eve_interception_count = 0
        if self.eve_enabled:
            eve_actions = self.step3_eve_interception(alice_bits, alice_bases)
            eve_interception_count = eve_actions['interception_count']
        
        # Step 4: Bob measurement
        bob_bases, bob_results = self.step4_bob_measurement(alice_bits, alice_bases, eve_actions)
        
        # Step 5: Basis reconciliation
        matching_indices, sifted_key_alice, sifted_key_bob = self.step5_basis_reconciliation(
            alice_bits, alice_bases, bob_bases, bob_results
        )
        
        # Step 6: QBER calculation
        qber, errors, sample_size, eve_detected, status = self.step6_qber_calculation(
            alice_bits, bob_results, matching_indices
        )
        
        return {
            'alice_bits': alice_bits,
            'alice_bases': alice_bases,
            'encoded_qubits': encoded_qubits,
            'eve_actions': eve_actions,
            'bob_bases': bob_bases,
            'bob_results': bob_results,
            'matching_indices': matching_indices,
            'sifted_key_alice': sifted_key_alice,
            'sifted_key_bob': sifted_key_bob,
            'qber': round(qber, 2),
            'errors': errors,
            'sample_size': sample_size,
            'eve_detected': eve_detected,
            'status': status,
            'eve_interception_count': eve_interception_count
        }


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "BB84 Quantum Key Distribution API",
        "version": "1.0.0",
        "endpoints": {
            "simulate": "/bb84/simulate"
        }
    }


@app.post("/bb84/simulate", response_model=BB84Response)
async def simulate_bb84(request: BB84Request):
    """
    Simulate BB84 Quantum Key Distribution protocol
    
    Args:
        request: BB84Request with num_bits, eve_enabled, and eve_interception_rate
    
    Returns:
        BB84Response with complete protocol results including QBER analysis
    """
    try:
        protocol = BB84Protocol(
            num_bits=request.num_bits,
            eve_enabled=request.eve_enabled,
            eve_interception_rate=request.eve_interception_rate
        )
        
        results = protocol.run_protocol()
        return BB84Response(**results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BB84 QKD API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)