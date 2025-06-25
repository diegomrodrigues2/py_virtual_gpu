# Contribuindo

## Setup de desenvolvimento

1. Crie um ambiente virtual opcionalmente e instale o pacote no modo editável com as dependências opcionais definidas em `[project.optional-dependencies]` do `pyproject.toml`:

```bash
pip install -e .[api]
```

2. Instale também os pacotes listados em `requirements.txt`:

```bash
pip install -r requirements.txt
```

3. Execute a suíte de testes na raiz do repositório:

```bash
pytest
```


