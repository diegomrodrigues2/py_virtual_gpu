# Componentes e Funcionamento

Este documento fornece uma visão resumida dos principais componentes da GPU virtual e como eles interagem durante a execução de um *kernel*.

## VirtualGPU

O objeto `VirtualGPU` representa o dispositivo inteiro. Ele agrega diversos `StreamingMultiprocessor`s (SMs) e a `GlobalMemory`. O papel do dispositivo é dividir a grade de execução em `ThreadBlock`s e distribuí-los entre os SMs.

## StreamingMultiprocessor

Cada `StreamingMultiprocessor` possui sua própria `SharedMemory` e uma fila de `ThreadBlock`s. Um SM executa os blocks de forma sequencial ou em round-robin, criando `Warp`s para conduzir as `Thread`s em *lock-step*.

## ThreadBlock

Um `ThreadBlock` é um agrupamento de threads que partilham uma região de `SharedMemory` e podem se sincronizar por meio de uma barreira. Ao ser despachado para um SM, o block instancia suas threads e inicia a execução do kernel.

## Thread e Warp

`Thread` é a menor unidade de execução. Ela possui registradores privados e referências para as memórias compartilhada e global. As threads são agrupadas em `Warp`s para simular o modelo SIMT. O warp controla a máscara de threads ativas e lida com possível divergência de fluxo.

## GlobalMemory e SharedMemory

`GlobalMemory` é acessível por todos os SMs e blocks. Ela é implementada com `multiprocessing.Array` para permitir o compartilhamento entre processos. Já `SharedMemory` é restrita a um block ou SM e é usada para comunicação rápida entre suas threads.

## Fluxo de Execução

1. O usuário chama `launch_kernel` na `VirtualGPU`, indicando a dimensão do grid e do block.
2. A GPU virtual cria os `ThreadBlock`s necessários e os coloca nas filas de execução dos SMs.
3. Cada SM consome sua fila, dividindo os blocks em `Warp`s que executam em *lock-step*.
4. Durante a execução, as threads acessam `SharedMemory` e `GlobalMemory` conforme definido pelo kernel.
5. Após a conclusão de todos os blocks, a execução termina e os resultados podem ser copiados de volta para a aplicação hospedeira.


## Funcionalidades de Memoria e Execucao

- `VirtualGPU.malloc` e `VirtualGPU.free` permitem gerenciar espacos em `GlobalMemory` por meio de `DevicePointer`.
- `memcpy_host_to_device`, `memcpy_device_to_host` e `memcpy` copiam dados entre CPU e GPU.
- O decorador `@kernel` transforma funcoes Python em kernels e utiliza o dispositivo definido por `VirtualGPU.set_current`.
- `launch_kernel` divide o grid em `ThreadBlock`s e distribui entre os SMs, expondo `threadIdx`, `blockIdx`, `blockDim` e `gridDim` para o kernel.
