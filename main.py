from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import random
import time

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


class RSARequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=100, description="Message to encrypt")
    eve_enabled: bool = Field(..., description="Enable eavesdropper")
    simulation_speed: float = Field(default=1.0, ge=0.5, le=3.0, description="Animation speed multiplier")
    key_size: int = Field(default=16, ge=8, le=32, description="Key size in bits (for educational simulation)")


class RSAAnimationStep(BaseModel):
    stage: str
    timestamp: float
    message_state: Dict[str, Any]
    eve_state: Optional[Dict[str, Any]]
    description: str


class RSAResponse(BaseModel):
    message: str
    encrypted_message: str
    encrypted_numbers: List[int]
    public_key: Dict[str, int]  # {e, n}
    private_key: Dict[str, int]  # {d, n}
    prime_p: int
    prime_q: int
    n: int
    phi_n: int
    encryption_steps: List[Dict[str, Any]]
    decryption_steps: List[Dict[str, Any]]
    eve_enabled: bool
    eve_intercepted: bool
    eve_copied_data: Optional[str]
    animation_steps: List[RSAAnimationStep]
    metrics: Dict[str, Any]
    status: str


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
            "bb84_simulate": "/bb84/simulate",
            "rsa_simulate": "/rsa/simulate"
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


class RSASimulation:
    """Comprehensive RSA encryption simulation with full mathematical workflow"""
    
    def __init__(self, message: str, eve_enabled: bool = False, key_size: int = 16):
        self.message = message
        self.eve_enabled = eve_enabled
        self.key_size = key_size
        self.animation_steps = []
        
    def is_prime(self, n: int) -> bool:
        """Check if a number is prime"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def generate_prime(self, bits: int) -> int:
        """Generate a random prime number with specified bit length"""
        min_value = 2 ** (bits - 1)
        max_value = 2 ** bits - 1
        
        # Try to find a prime within reasonable attempts
        for _ in range(1000):
            candidate = random.randint(min_value, max_value)
            if self.is_prime(candidate):
                return candidate
        
        # Fallback to smaller primes if generation fails
        small_primes = [61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137]
        return random.choice(small_primes)
    
    def gcd(self, a: int, b: int) -> int:
        """Calculate greatest common divisor"""
        while b:
            a, b = b, a % b
        return a
    
    def extended_gcd(self, a: int, b: int):
        """Extended Euclidean Algorithm"""
        if a == 0:
            return b, 0, 1
        gcd_val, x1, y1 = self.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd_val, x, y
    
    def mod_inverse(self, e: int, phi: int) -> int:
        """Calculate modular multiplicative inverse"""
        gcd_val, x, _ = self.extended_gcd(e, phi)
        if gcd_val != 1:
            raise Exception("Modular inverse does not exist")
        return (x % phi + phi) % phi
    
    def generate_keys(self):
        """
        Complete RSA key generation with all mathematical steps
        Returns: (public_key, private_key, p, q, n, phi_n, e, d)
        """
        # Step 1: Generate two distinct prime numbers
        p = self.generate_prime(self.key_size // 2)
        q = self.generate_prime(self.key_size // 2)
        
        # Ensure p and q are different
        while p == q:
            q = self.generate_prime(self.key_size // 2)
        
        # Step 2: Calculate n = p * q
        n = p * q
        
        # Step 3: Calculate Euler's totient function φ(n) = (p-1)(q-1)
        phi_n = (p - 1) * (q - 1)
        
        # Step 4: Choose public exponent e
        # Commonly e = 65537, but for small numbers we'll use smaller values
        e = 65537 if phi_n > 65537 else 3
        
        # Find e that is coprime with phi_n
        while self.gcd(e, phi_n) != 1:
            e += 2
            if e >= phi_n:
                e = 3
        
        # Step 5: Calculate private exponent d
        # d is the modular multiplicative inverse of e modulo φ(n)
        d = self.mod_inverse(e, phi_n)
        
        public_key = {'e': e, 'n': n}
        private_key = {'d': d, 'n': n}
        
        return public_key, private_key, p, q, n, phi_n, e, d
    
    def encrypt_char(self, char: str, e: int, n: int) -> int:
        """Encrypt a single character using RSA formula: c = m^e mod n"""
        m = ord(char)  # Convert character to ASCII value
        c = pow(m, e, n)  # Modular exponentiation
        return c
    
    def decrypt_number(self, c: int, d: int, n: int) -> str:
        """Decrypt a single number using RSA formula: m = c^d mod n"""
        m = pow(c, d, n)  # Modular exponentiation
        return chr(m)  # Convert ASCII value back to character
    
    def encrypt_message(self, message: str, public_key: Dict[str, int]) -> tuple:
        """
        Encrypt entire message character by character
        Returns: (encrypted_numbers, encryption_steps)
        """
        e = public_key['e']
        n = public_key['n']
        encrypted_numbers = []
        encryption_steps = []
        
        for i, char in enumerate(message):
            m = ord(char)
            c = self.encrypt_char(char, e, n)
            encrypted_numbers.append(c)
            
            encryption_steps.append({
                'char': char,
                'ascii': m,
                'formula': f"{m}^{e} mod {n}",
                'encrypted': c,
                'step_number': i + 1
            })
        
        return encrypted_numbers, encryption_steps
    
    def decrypt_message(self, encrypted_numbers: List[int], private_key: Dict[str, int]) -> tuple:
        """
        Decrypt entire message number by number
        Returns: (decrypted_message, decryption_steps)
        """
        d = private_key['d']
        n = private_key['n']
        decrypted_chars = []
        decryption_steps = []
        
        for i, c in enumerate(encrypted_numbers):
            char = self.decrypt_number(c, d, n)
            m = ord(char)
            decrypted_chars.append(char)
            
            decryption_steps.append({
                'encrypted': c,
                'formula': f"{c}^{d} mod {n}",
                'ascii': m,
                'char': char,
                'step_number': i + 1
            })
        
        return ''.join(decrypted_chars), decryption_steps
    
    def run_simulation(self):
        """Execute complete RSA simulation with detailed mathematical steps"""
        start_time = time.time()
        
        # Step 1: Key Generation
        public_key, private_key, p, q, n, phi_n, e, d = self.generate_keys()
        
        self.animation_steps.append({
            'stage': 'key-generation',
            'timestamp': time.time() - start_time,
            'message_state': {
                'text': self.message,
                'encrypted': False,
                'position': 0
            },
            'eve_state': None,
            'description': f'Bob generates RSA key pair using primes p={p}, q={q}'
        })
        
        # Step 2: Encryption
        time.sleep(0.01)
        encrypted_numbers, encryption_steps = self.encrypt_message(self.message, public_key)
        encrypted_display = ''.join(['█' for _ in self.message])
        
        self.animation_steps.append({
            'stage': 'encrypting',
            'timestamp': time.time() - start_time,
            'message_state': {
                'text': encrypted_display,
                'encrypted': True,
                'position': 0
            },
            'eve_state': None,
            'description': f'Alice encrypts "{self.message}" using public key (e={e}, n={n})'
        })
        
        # Step 3: Transmission with optional Eve interception
        time.sleep(0.01)
        eve_intercepted = False
        eve_copied_data = None
        
        for position in [0, 25, 50, 75, 100]:
            stage = 'transmitting'
            description = f'Encrypted message in transit ({position}%)'
            
            # Eve intercepts at 50%
            if self.eve_enabled and position == 50:
                stage = 'eve-intercepting'
                eve_intercepted = True
                eve_copied_data = str(encrypted_numbers)
                description = 'Eve intercepts and copies encrypted message (UNDETECTED!)'
            
            self.animation_steps.append({
                'stage': stage,
                'timestamp': time.time() - start_time,
                'message_state': {
                    'text': encrypted_display,
                    'encrypted': True,
                    'position': position
                },
                'eve_state': {
                    'intercepted': eve_intercepted,
                    'copied_data': eve_copied_data if eve_intercepted else None,
                    'can_decrypt': False
                } if eve_intercepted and position == 50 else None,
                'description': description
            })
            time.sleep(0.005)
        
        # Step 4: Decryption
        time.sleep(0.01)
        decrypted_message, decryption_steps = self.decrypt_message(encrypted_numbers, private_key)
        
        self.animation_steps.append({
            'stage': 'decrypting',
            'timestamp': time.time() - start_time,
            'message_state': {
                'text': decrypted_message,
                'encrypted': False,
                'position': 100
            },
            'eve_state': {'intercepted': eve_intercepted} if eve_intercepted else None,
            'description': f'Bob decrypts message using private key (d={d}, n={n})'
        })
        
        # Step 5: Complete
        self.animation_steps.append({
            'stage': 'complete',
            'timestamp': time.time() - start_time,
            'message_state': {
                'text': decrypted_message,
                'encrypted': False,
                'position': 100
            },
            'eve_state': {'intercepted': eve_intercepted} if eve_intercepted else None,
            'description': 'Transmission complete - Eve\'s presence UNKNOWN!' if eve_intercepted else 'Transmission complete successfully'
        })
        
        # Calculate metrics
        metrics = {
            'error_rate': 0.0,
            'eve_detection_rate': 0.0,
            'key_efficiency': 100.0,
            'quantum_safe': False,
            'transmission_time': time.time() - start_time
        }
        
        return {
            'message': self.message,
            'encrypted_message': encrypted_display,
            'encrypted_numbers': encrypted_numbers,
            'public_key': public_key,
            'private_key': private_key,
            'prime_p': p,
            'prime_q': q,
            'n': n,
            'phi_n': phi_n,
            'encryption_steps': encryption_steps,
            'decryption_steps': decryption_steps,
            'eve_enabled': self.eve_enabled,
            'eve_intercepted': eve_intercepted,
            'eve_copied_data': eve_copied_data,
            'animation_steps': self.animation_steps,
            'metrics': metrics,
            'status': 'Success - Message delivered (Eve presence UNKNOWN)' if eve_intercepted else 'Success - Message delivered'
        }


@app.post("/rsa/simulate", response_model=RSAResponse)
async def simulate_rsa(request: RSARequest):
    """
    Simulate RSA Classical Encryption with full mathematical workflow
    
    Args:
        request: RSARequest with message, eve_enabled, simulation_speed, and key_size
    
    Returns:
        RSAResponse with complete encryption process including:
        - Prime number selection (p, q)
        - Key generation steps (n, φ(n), e, d)
        - Character-by-character encryption details
        - Character-by-character decryption details
        - Animation steps for visualization
    """
    try:
        simulation = RSASimulation(
            message=request.message,
            eve_enabled=request.eve_enabled,
            key_size=request.key_size
        )
        
        results = simulation.run_simulation()
        return RSAResponse(**results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BB84 QKD API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)