class AtividadeHistorica(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(default='', max_length=1200)
    activity_date: Optional[str] = None


class PedidoRF08(BaseModel):
    ramo: str = Field(min_length=2, max_length=100)
    tema: str = Field(min_length=3, max_length=120)
    historico_atividades: List[AtividadeHistorica] = Field(
        default_factory=list,
        max_length=20
    )


class AtividadeSugerida(BaseModel):
    title: str
    description: str


class RespostaRF08(BaseModel):
    atividades: List[AtividadeSugerida]