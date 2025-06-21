# Estrutura de Classes

Este documento resume as responsabilidades e a interface básica das principais classes da GPU virtual. Para uma visão geral do mapeamento entre os elementos do modelo de programação GPU e as estruturas simuladas consulte a **Tabela 2** em [RESEARCH.md](../RESEARCH.md).

A implementação detalhada destas classes e de suas APIs será tratada nas **issues 2.1 a 3.3** do repositório.

## `VirtualGPU`

**Responsabilidade:** representa o dispositivo de execução paralelo completo, contendo os `StreamingMultiprocessor`s (SMs) e a memória global. É responsável por coordenar o lançamento de *kernels* e o gerenciamento da memória.

**Principais atributos**
- `sms`: lista dos SMs simulados disponíveis.
- `global_memory`: área de memória acessível por todos os blocks e threads.

**Métodos principais**
- `launch_kernel(func, grid_dim, block_dim, *args)`: divide a grade em blocks, agenda sua execução nos SMs e repassa os argumentos para cada thread.
- `malloc(size) / free(ptr)`: interface para alocação e liberação de espaço em `global_memory`.
- `memcpy_host_to_device(...)` e `memcpy_device_to_host(...)`: simulam transferências entre CPU e GPU.

## `StreamingMultiprocessor`

**Responsabilidade:** simula um SM, agendando `ThreadBlock`s e gerenciando a execução das threads em cada block.

**Principais atributos**
- `blocks`: lista de `ThreadBlock`s em execução.
- `shared_memory`: memória on-chip de acesso rápido utilizada pelos blocks em execução.

**Métodos principais**
- `execute_block(block)`: executa um `ThreadBlock`, controlando a sincronização interna e o andamento das threads.
- `schedule(blocks)`: lógica de distribuição de blocks atribuídos a este SM.

## `ThreadBlock`

**Responsabilidade:** representa um grupo de threads que compartilham memória e podem se sincronizar.

**Principais atributos**
- `threads`: coleção de `Thread`s pertencentes ao block.
- `shared_memory`: área de memória acessível somente pelas threads deste block.
- `barrier`: mecanismo de sincronização (ex.: `multiprocessing.Barrier`).

**Métodos principais**
- `run()`: inicia a execução de todas as threads do block.
- `syncthreads()`: faz todas as threads aguardarem na barreira, imitando `__syncthreads()`.

## `Thread`

**Responsabilidade:** menor unidade de execução do kernel, contendo registradores privados e os índices de thread e block.

**Principais atributos**
- `registers`: memória privada de cada thread.
- `thread_idx` e `block_idx`: índices que identificam a posição desta thread no grid.

**Métodos principais**
- `execute(kernel_func, *args)`: executa a função do kernel com os argumentos fornecidos.
- `read_write_memory(...)`: operações utilitárias para acessar a memória global ou compartilhada conforme necessário.

