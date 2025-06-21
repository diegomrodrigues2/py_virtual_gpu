### Objetivo do Projeto

Este projeto visa desenvolver uma **GPU Virtual completa em Python** para fins educacionais e de aprendizagem profunda sobre arquitetura de computadores paralelos. O simulador replica fielmente os componentes centrais de uma GPU moderna (Streaming Multiprocessors, hierarquia de memória, modelo SIMT) e seu modelo de programação (kernels, grids, blocks, threads), utilizando `multiprocessing` para superar o GIL do Python e alcançar paralelismo verdadeiro.

**Objetivos Principais:**
- **Aprendizagem Prática**: Desmistificar a computação por GPU através de simulação funcional interativa, eliminando a necessidade de hardware caro
- **Precisão Conceitual**: Focar na correção funcional e fidelidade ao modelo de programação CUDA/OpenCL, priorizando clareza educacional sobre precisão de ciclo
- **Paralelismo de Dados**: Demonstrar como cargas de trabalho são distribuídas e processadas em paralelo, incluindo cenários multi-GPU
- **Ambiente Seguro**: Proporcionar uma plataforma controlada para experimentação com conceitos de GPU sem riscos ao hardware

O simulador permite visualizar concretamente como abstrações de alto nível (kernels, synchronization, memory hierarchy) se traduzem em operações de hardware subjacentes, preenchendo a lacuna entre teoria e prática na programação paralela.

---

## Organização das Issues

### 1. Estruturação Inicial e Arquitetura

- **1.3** Esboço da Estrutura de Classes Principais (VirtualGPU, SM, Block, Thread)

---

### 2. Implementação dos Componentes Básicos

- **2.1** Implementação da Classe VirtualGPU (estrutura básica)
- **2.2** Implementação da GlobalMemory usando multiprocessing.Array
- **2.3** Implementação da Classe StreamingMultiprocessor (SM) (estrutura básica)
- **2.4** Implementação da Classe ThreadBlock (estrutura básica)
- **2.5** Implementação da Classe Thread com RegisterMemory (privada)
- **2.6** Implementação da SharedMemory dentro de ThreadBlock (comunicação interna)

---

### 3. Modelo de Execução e API

- **2.7** Simulação do Modelo SIMT: Grupos de warp e execução sequencial/paralela simulada
- **2.8** Mecanismo de Despacho de Instruções e Divergência de Warp (conceitual)
- **3.1** Desenvolvimento da API de Gerenciamento de Memória (malloc, free)
- **3.2** Implementação da API de Transferência de Dados (memcpy_host_to_device, device_to_host)
- **3.3** Desenvolvimento do Decorador @kernel para funções Python
- **3.4** Implementação do Lançamento de Kernel (launch_kernel) e Contexto de Execução
- **3.5** Acesso aos Índices de Thread/Block e Dimensões dentro do Kernel

---

### 4. Execução Paralela e Sincronização

- **4.1** Integração com multiprocessing.Pool para Agendamento de ThreadBlocks
- **4.2** Implementação de Sincronização por Barreira (multiprocessing.Barrier) no ThreadBlock
- **4.3** Implementação de Operações Atômicas Simuladas (com Lock/Value)
- **4.4** Refinamento do Modelo de Divergência de Warp (monitoramento e visualização conceitual)

---

### 5. Multi-GPU e Escalabilidade

- **5.1** Adaptação para Múltiplas Instâncias de VirtualGPU (Processos Independentes)
- **5.2** Implementação de Estratégias de Distribuição de Dados para Multi-GPU
- **5.3** Simulação de Comunicação Inter-GPU (via Queue/Pipe)
- **5.5** Testes de Escalabilidade com Múltiplos SMs e VirtualGPUs

---

### 6. Ferramentas de Suporte e Monitoramento

- **6.1** Implementação de Sistema de Logging Detalhado e Rastreamento de Eventos
- **6.2** Desenvolvimento de Contadores de Desempenho Conceituais (acessos à memória, divergências)
- **6.3** Ferramentas Básicas de Depuração (inspeção de estado de memória/registradores)

---

**Link para as issues:**  
https://github.com/diegomrodrigues2/py_virtual_gpu/issues
