# Technical notes from the safe-6 implementation

## Что ускорили и почему

Большой словарь создаёт крупные промежуточные logits, decoder удерживает активации и обрабатывает padding, а NF4 требует восстановления весов. Мы сократили объём промежуточных тензоров и обмен с памятью. GEMM остаются в cuBLAS, attention в torch SDPA.

### Head: порционная обработка logits

[`solution/head/`](../solution/head/) обрабатывает по 256 токеновых строк. Вместо полной матрицы N×V повторно используется fp16-буфер `[256, 151936]`, **74 MiB**. `head_row_softmax_kernel` считает online softmax, entropy, target logit и first-index argmax. Для взвешенных строк он перезаписывает logits градиентом dlogits. cuBLAS вычисляет dX в forward, backward использует готовый dX.

Torch формирует loss и accuracy из статистик. Потери строк остаются доступны Soup rewind. Замороженные tied embedding/head хранят одни исходные fp16-значения, без дубликата fp32-весов. Выход embedding остаётся fp32. Новых обучаемых параметров нет.

### Decoder: padding, слияние операций и пересчёт

[`solution/body/packed.py`](../solution/body/packed.py) собирает реальные токены **внутри существующего batch**, сохраняя строки, позиции и порядок. Causal SDPA выполняется отдельно для каждой исходной строки, результат возвращается в исходную раскладку. Строки разных batch не смешиваются.

Кернел residual + RMSNorm объединяет чтения памяти. Отдельный RoPE-кернел добавляет LoRA-ветку q. Подходящие нормализации сохраняют fp16-выход и fp32 inverse standard deviation, backward восстанавливает xhat. Для проблемных каналов есть точные fp32 side saves, для layer-0 и final norm сохраняется residual. Backward восстанавливает dropout-маску LoRA и пересчитывает промежуточные gate/up MLP.

### NF4: восстановление масштабов и весов за один проход

[`solution/nf4/`](../solution/nf4/) декодирует nested scales и упакованные NF4-веса. Grouped decoder записывает q/k/v в общий fp16-буфер перед одним cuBLAS GEMM. Backward повторно декодирует веса для dX, поэтому плотная копия не живёт до конца backward.

Декодированные веса на поддержанном пути bit-exact. Результаты объединённых проекций и обучение проверяются с допуском: выбор GEMM, порядок накопления и fp16-реконструкция могут менять округление. Dropout сохраняет вероятность, масштаб и место применения, но не индивидуальные маски базы.

### Подготовка процесса

`apply()` запускает JIT-компиляцию в потоке этого же процесса параллельно загрузке модели. Первый training step и выход из процесса ожидают завершения потока. Компиляция входит в измеряемое время, готовых кэшей между запусками нет. Startup-патчи уменьшают повторное сканирование пакетов, GC и стоимость dtype cast при загрузке, меняют fingerprints только для dataset без файлового кэша. Состав, порядок данных и Soup rewind fingerprint сохраняются.

Флаг safe-6 `store_binary_only` сокращает записи промежуточного Triton IR. Отдельный причинный выигрыш этого флага на T4 не установлен, поэтому мы не приписываем ему итоговое ускорение.

## Наши Triton entry point

| Запускаемый кернел | Исходный файл |
|---|---|
| `nf4_dequant_kernel` | [solution/nf4/dequant.py](../solution/nf4/dequant.py) |
| `nf4_dequant_multi_kernel` | [solution/nf4/dequant.py](../solution/nf4/dequant.py) |
| `_add_rmsnorm_fwd` | [solution/body/norm_rope.py](../solution/body/norm_rope.py) |
| `_add_rmsnorm_bwd` | [solution/body/norm_rope.py](../solution/body/norm_rope.py) |
| `_rope_kernel` | [solution/body/norm_rope.py](../solution/body/norm_rope.py) |
| `lora_dropout_fwd_kernel` | [solution/body/lora.py](../solution/body/lora.py) |
| `lora_dropout_bwd_kernel` | [solution/body/lora.py](../solution/body/lora.py) |
| `head_row_softmax_kernel` | [solution/head/kernels.py](../solution/head/kernels.py) |

Все восемь выполняют полезную работу в каждом измеренном solution run. Inlined helpers не считаются отдельными запусками. Неизменённый пример организатора `solution/swiglu.py` явно исключён из количества и доли собственных кернелов. Готовые kernel-библиотеки не подключены.

## Ограничения и неудачные эксперименты

Решение специализируется на закреплённой Qwen2/fp16/LoRA конфигурации и версиях библиотек. Для неподдержанных входов предусмотрены guarded fallbacks. Численная эквивалентность всего обучения проверяется с допуском. Пересчёт MLP экономит память ценой дополнительных вычислений, и его эффект зависит от задачи.

Публичный прогон не доказывает результат на скрытых строках или других GPU. Разброс T4 и частоты карты влияют на время. Распределение общего выигрыша между отдельными патчами в финальном запуске не измерено.

Поздний saved-activation кандидат TR1 не прошёл внутренние speed screens и потребовал больше памяти. Кандидат b76 прошёл локальную проверку, но не имеет завершённого полного замера T4. Поэтому финальным остаётся safe-6. Activation offload и NF4 streaming в него не входят. Старые незавершённые baseline-OOM эксперименты не выдаются за сравнительные победы. Ранние 82,39 / 49,25 с из handoff ZIP являются историческим запуском и не смешиваются с результатами выше.

## Источники идей

Исходники содержат подробные атрибуции. Ниже источники идей, а не готовые kernel-зависимости или скопированный код:

- Head: [Liger Kernel](https://github.com/linkedin/Liger-Kernel), [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009), [online softmax](https://arxiv.org/abs/1805.02867).
- Decoder: [RMSNorm](https://arxiv.org/abs/1910.07467), [RoPE](https://arxiv.org/abs/2104.09864), [Apex PR 1715](https://github.com/NVIDIA/apex/pull/1715), [Triton low-memory dropout](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html), [unpadded layouts](https://huggingface.co/blog/packing-with-FA2), [activation recomputation](https://arxiv.org/abs/1604.06174).
- NF4 и проекции: [QLoRA](https://arxiv.org/abs/2305.14314), [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [паттерны LoRA/MLP Unsloth](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py), fused projections Hugging Face.
- JIT warmup: API и исходники Triton 3.6 `JITFunction.warmup`, `CompiledKernel._init_handles`. GEMM оставлены в cuBLAS с учётом наблюдённого lowering Triton 3.6 на sm_75. У T4 есть tensor cores.

Лицензия кода: Apache-2.0, сохранены [LICENSE](../LICENSE) и [NOTICE](../NOTICE). Данные [AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2): MIT на карточке, казахский перевод [Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), CC BY-NC 4.0. Модель [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct): Apache-2.0. Данные и модель загружаются во время подготовки.

