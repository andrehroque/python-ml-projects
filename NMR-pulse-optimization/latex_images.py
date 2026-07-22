from qiskit import QuantumCircuit

qc = QuantumCircuit(3)

qc.h(0)
qc.cswap(0,1,2)
qc.h(0)

print(qc.draw('latex_source'))
qc.draw()