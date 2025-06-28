# Componentes e Funcionamento

Este documento fornece uma visão resumida dos principais componentes da GPU virtual e como eles interagem durante a execução de um *kernel*.

## VirtualGPU

O objeto `VirtualGPU` representa o dispositivo inteiro. Ele agrega diversos `StreamingMultiprocessor`s (SMs) e a `GlobalMemory`. O papel do dispositivo é dividir a grade de execução em `ThreadBlock`s e distribuí-los entre os SMs.

## StreamingMultiprocessor

Cada `StreamingMultiprocessor` possui sua própria `SharedMemory` e uma fila de `ThreadBlock`s. Um SM executa os blocks de forma sequencial ou em round-robin, criando `Warp`s para conduzir as `Thread`s em *lock-step*. O método `dispatch()` emite uma instrução por warp a cada "ciclo" e reenfileira aqueles que continuam ativos. O contador `warps_executed` é incrementado a cada emissão.

## ThreadBlock

Um `ThreadBlock` é um agrupamento de threads que partilham uma região de `SharedMemory` e podem se sincronizar por meio de uma barreira. Ao ser despachado para um SM, o block instancia suas threads e inicia a execução do kernel.

## Thread e Warp

`Thread` é a menor unidade de execução. Ela possui registradores privados e referências para as memórias compartilhada e global. As threads são agrupadas em `Warp`s para simular o modelo SIMT. O warp controla a máscara de threads ativas e lida com possível divergência de fluxo.

## GlobalMemory e SharedMemory

`GlobalMemory` é acessível por todos os SMs e blocks. Ela é implementada com `multiprocessing.Array` para permitir o compartilhamento entre processos. Já `SharedMemory` é restrita a um block ou SM e é usada para comunicação rápida entre suas threads.

## Hierarquia de Memória

Os diferentes níveis da hierarquia são representados por subclasses de
``MemorySpace``. Cada uma possui latência e largura de banda conceituais que são
acumuladas nos campos ``stats`` sempre que ``read`` ou ``write`` são chamados.
Entre elas estão ``RegisterFile`` (registradores privados), ``SharedMemory``
(on-chip), ``L1Cache``/``L2Cache`` (caches), ``GlobalMemorySpace`` e as regiões
especializadas ``ConstantMemory`` e ``LocalMemory``. O método
``reset_stats()`` permite zerar os contadores para novas medições.

### ConstantMemory

``ConstantMemory`` é uma área de somente leitura (``64 KiB`` por padrão)
compartilhada por todas as threads. O host copia valores usando
``gpu.set_constant`` e os kernels podem ler via ``thread.const_mem.read`` ou
``VirtualGPU.get_current().read_constant``.

```python
gpu.set_constant(b"ola")                     # host -> constant memory
value = VirtualGPU.get_current().read_constant(0, 3)
```

## Fluxo de Execução

1. O usuário chama `launch_kernel` na `VirtualGPU`, indicando a dimensão do grid e do block.
2. A GPU virtual cria os `ThreadBlock`s necessários e os coloca nas filas de execução dos SMs.
3. Cada SM consome sua fila, dividindo os blocks em `Warp`s que executam em *lock-step*.
4. Durante a execução, as threads acessam `SharedMemory` e `GlobalMemory` conforme definido pelo kernel.
5. Após a conclusão de todos os blocks, a execução termina e os resultados podem ser copiados de volta para a aplicação hospedeira.


## Funcionalidades de Memoria e Execucao

- `VirtualGPU.malloc` e `VirtualGPU.free` permitem gerenciar espacos em `GlobalMemory` por meio de `DevicePointer`.
- `memcpy_host_to_device`, `memcpy_device_to_host` e `memcpy` copiam dados entre CPU e GPU.
- Exemplo de cópia:
```python
ptr = gpu.malloc(256)
gpu.memcpy_host_to_device(b"\x00" * 256, ptr)
data = gpu.memcpy_device_to_host(ptr, 256)
```
- Os ponteiros retornados por `malloc` são instâncias de `DevicePointer` que
  aceitam aritmética e indexação (`ptr + n`, `ptr[i]`, etc.), permitindo sintaxe
  semelhante ao CUDA C++. Abaixo um exemplo de kernel que multiplica vetores
  utilizando `ptr[i]`:

```python
@kernel(grid_dim=(1, 1, 1), block_dim=(4, 1, 1))
def vec_mul(threadIdx, blockIdx, blockDim, gridDim, a_ptr, b_ptr, out_ptr):
    i = threadIdx[0]
    a = int.from_bytes(a_ptr[i], "little")
    b = int.from_bytes(b_ptr[i], "little")
    out_ptr[i] = (a * b).to_bytes(4, "little")
```
- O decorador `@kernel` transforma funcoes Python em kernels e utiliza o dispositivo definido por `VirtualGPU.set_current`.
- `launch_kernel` divide o grid em `ThreadBlock`s e distribui entre os SMs, expondo `threadIdx`, `blockIdx`, `blockDim` e `gridDim` para o kernel.
- `ThreadBlock.barrier_sync()` permite que as threads de um block aguardem umas
  às outras em um ponto comum, espelhando o comportamento de
  ``__syncthreads()`` do CUDA.
- A classe `SharedMemory` expõe operações atômicas (`atomic_add`, `atomic_sub`,
  `atomic_cas`, `atomic_max`, `atomic_min`, `atomic_exchange`) que permitem
  atualização segura de valores compartilhados entre threads.

## Fences de Memória

Três funções simulam as barreiras de memória do CUDA para controlar a
visibilidade de escritas:

- `threadfence_block()` garante que dados gravados na `SharedMemory` do bloco
  fiquem visíveis aos demais threads do mesmo block após o fence.
- `threadfence()` propaga atualizações para todas as memórias acessíveis dentro
  do dispositivo, cobrindo tanto a `SharedMemory` quanto a `GlobalMemory`.
- `threadfence_system()` estende o efeito para que a aplicação hospedeira possa
  observar as escritas; na simulação ele é equivalente a `threadfence()`.

Em todos os casos essas funções apenas adquirem e liberam os locks das
respectivas memórias para emular o efeito de ordenação.
## Operações Atômicas

A biblioteca também fornece *helpers* de alto nível para realizar operações atômicas diretamente sobre ``DevicePointer`` na ``GlobalMemory``. As funções ``atomicAdd``, ``atomicSub``, ``atomicCAS``, ``atomicMax``, ``atomicMin`` e ``atomicExchange`` recebem o ponteiro e o valor desejado, retornando o valor anterior.

```python
from py_virtual_gpu import kernel, atomicAdd

@kernel(grid_dim=(1, 1, 1), block_dim=(4, 1, 1))
def incr(threadIdx, blockIdx, blockDim, gridDim, counter_ptr):
    atomicAdd(counter_ptr, 1)
```

Esses *helpers* utilizam internamente ``GlobalMemory.atomic_*``. Os métodos
originais continuam acessíveis diretamente através das instâncias de
``SharedMemory`` e ``GlobalMemory``.


## Monitoramento de Divergência

Com o novo fluxo de execução cada warp é avançado em *lock-step*. O método
``dispatch()`` do SM seleciona um warp na fila e chama ``warp.execute()`` para
emitir uma instrução. Durante essa etapa a instrução é buscada, o predicado de
ramo é avaliado para todas as threads e a ``SIMTStack`` é atualizada para tratar
reconvergências. Sempre que a máscara de threads muda ``record_divergence``
armazena um ``DivergenceEvent`` em ``divergence_log`` e incrementa
``counters['warp_divergences']``. O total de instruções emitidas é acumulado em
``counters['warps_executed']`` e quaisquer ciclos extras provenientes de acesso
à memória são somados em ``stats['extra_cycles']``.

O log pode ser consultado para análise posterior. O exemplo a seguir constrói um
gráfico simples mostrando o número acumulado de divergências ao longo do ``pc``
de cada evento:

```python
log = sm.get_divergence_log()
pcs = [e.pc for e in log]
divs = list(range(1, len(log) + 1))
plt.plot(pcs, divs)
plt.xlabel("PC")
plt.ylabel("Divergências acumuladas")
plt.show()
print("Warps executados:", sm.counters["warps_executed"])
print("Divergências registradas:", sm.counters["warp_divergences"])
```

## Padrões de Acesso à Memória

O método ``Warp.memory_access`` permite registrar se um conjunto de endereços
é **coalescido** e se há **conflitos de banco** na ``SharedMemory``. Quando os
endereços não são contíguos, ``counters['non_coalesced_accesses']`` é
incrementado e ``stats['extra_cycles']`` recebe +1 ciclo conceitual. Conflitos de
banco retornados por ``SharedMemory.detect_bank_conflicts`` aumentam
``counters['bank_conflicts']`` e adicionam ``conflicts - 1`` ciclos extras.

```python
warp.memory_access([0, 8], 4)            # nao-coalesced
warp.memory_access([0, 0], 4, "shared")  # conflito de banco
coalescing = sm.report_coalescing_stats()
conflicts = sm.report_bank_conflict_stats()
print(coalescing)
print(conflicts)
```

## Spill de Registradores

Quando uma thread excede a capacidade de seu ``RegisterFile`` durante uma
escrita, os bytes extras são redirecionados para sua ``LocalMemory`` privada.
Cada spill registra eventos e adiciona ciclos extras calculados a partir de
``spill_granularity`` e ``spill_latency_cycles``. As estatísticas podem ser
obtidas por thread com ``thread.get_spill_stats()`` ou agregadas pela
``VirtualGPU.get_memory_stats()``.

## Memória Local por Thread

``LocalMemory`` é privada de cada ``Thread`` e possui a mesma latência da
``GlobalMemory``. Ela é usada para variáveis locais grandes e também para o
``spill`` automático dos registradores. A região é limitada e os kernels podem
reservar trechos com ``Thread.alloc_local``:

```python
@kernel(grid_dim=(1,1,1), block_dim=(1,1,1))
def exemplo(threadIdx, blockIdx, blockDim, gridDim):
    off = thread.alloc_local(4)
    thread.local_mem.write(off, b"data")
    valor = thread.local_mem.read(off, 4)
```

## Operações de Warp

Duas funções auxiliam a troca de informações entre as threads de um mesmo warp.
Elas utilizam a barreira compartilhada do bloco para orquestrar a sincronização.

- ``shfl_sync(valor, src_lane)`` devolve o valor fornecido pela lane
  ``src_lane`` para todas as threads:

```python
from py_virtual_gpu import kernel, shfl_sync

@kernel(grid_dim=(1,1,1), block_dim=(4,1,1))
def copia_primeira(threadIdx, blockIdx, blockDim, gridDim, out):
    v = threadIdx[0]
    out[threadIdx[0]] = shfl_sync(v, 0)
```

- ``ballot_sync(predicado)`` reúne um predicado de cada lane e retorna uma
  máscara de bits onde cada posição representa o resultado de uma thread:

```python
@kernel(grid_dim=(1,1,1), block_dim=(4,1,1))
def mascaras(threadIdx, blockIdx, blockDim, gridDim, out):
    m = ballot_sync(threadIdx[0] % 2 == 0)
    out[threadIdx[0]] = m
```

## Sincronização com Barreiras

O método ``syncthreads()`` expõe a barreira compartilhada do ``ThreadBlock``.
Todas as threads devem alcançar a chamada para que a execução continue. Se
apenas parte delas executa ``syncthreads()``, o bloco entra em *deadlock*.
Isso costuma acontecer quando há **divergência de warp** dentro de um ramo
condicional. Sempre garanta que as rotas de código reconvergem antes da
barreira ou mova a sincronização para fora do trecho divergente.

### Reduções e Scans

Operações de redução e de *scan* são exemplos clássicos que intercalam
escritas na ``SharedMemory`` com barreiras. O trecho abaixo mostra o padrão
de uma soma reduzida por bloco:

```python
from py_virtual_gpu import kernel, syncthreads
from py_virtual_gpu.thread import get_current_thread

@kernel(grid_dim=(1,1,1), block_dim=(8,1,1))
def reduce_sum(threadIdx, blockIdx, blockDim, gridDim, data):
    tx = threadIdx[0]
    shared = get_current_thread().shared_mem
    shared.write(tx * 4, data[tx])
    syncthreads()

    stride = blockDim[0] // 2
    while stride > 0:
        if tx < stride:
            a = int.from_bytes(shared.read(tx * 4, 4), "little")
            b = int.from_bytes(shared.read((tx + stride) * 4, 4), "little")
            shared.write(tx * 4, (a + b).to_bytes(4, "little"))
        syncthreads()
        stride //= 2
```

Um *scan* (prefix sum) segue lógica semelhante: cada iteração lê valores da
posição anterior, escreve o resultado e sincroniza para que todos observem os
intermediários antes do próximo passo.

### Atômicos ou Barreiras?

Use operações atômicas quando múltiplas threads precisam atualizar o mesmo
endereço de forma independente, como ao incrementar um contador global. Quando
os valores são acumulados cooperativamente em ``SharedMemory`` é preferível
empregar ``syncthreads()`` para sincronizar cada etapa. Caso os resultados
precisem ser visíveis além do bloco, utilize também ``threadfence_block()``,
``threadfence()`` ou ``threadfence_system()`` conforme o escopo de memória
necessário.
