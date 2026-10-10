# Развёртывание только Live Shadow Mode на Linux VPS

Публичная история документации очищена перед публикацией. Для ранее
установленного private deployment сначала прочитайте [замечание о Git
history](../docs/publication.md): автоматический fast-forward из публичного
репозитория может быть невозможен. Не делайте force/reset production.

AI Crypto Trader использует пример каталога `/opt/ai-crypto-trader/` и отдельного пользователя, virtualenv, `.env`, SQLite, журналы и systemd units. Остановка или обновление приложения не должны затрагивать другие сервисы сервера. Здесь работают только публичные `BTCUSDT spot 1h` свечи и локальный журнал shadow-решений; API-ключи не нужны. Никаких PAPER или Bybit ордеров эти unit-файлы не запускают.

Перед командами замените `operator@VPS` на свой SSH-логин и адрес сервера. Команды под заголовком «Mac» выполняются на Mac; остальные — только на VPS. **Codex не подключается к VPS и не переносит базу автоматически.** На сервере нужны Linux с systemd, Python 3.9+, Git и `sqlite3` CLI. Примеры рассчитаны на Debian/Ubuntu; для другого дистрибутива адаптируйте установку пакетов.

## Архитектура и поведение

- `ai-crypto-shadow.timer`: каждые 10 минут (`OnCalendar=*-*-* *:00/10:00`), без автоматического запуска за пропущенные тики после reboot (`Persistent=false`).
- `ai-crypto-shadow.service`: `Type=oneshot`, запускает только `/opt/ai-crypto-trader/.venv/bin/python -m ai_crypto_trader.diagnostics.shadow_runner --once`. Service не надо `enable` отдельно.
- Код, `.venv`, `.env` и БД находятся в `/opt/ai-crypto-trader/`; они не пересекаются с файлами других сервисов. `.env` доступен только пользователю `ai-crypto-shadow`, БД и файл блокировки — в `data/`. Логи находятся в systemd journal, отдельные файлы логов не создаются.
- CLI держит OS `flock` на `data/shadow_trading.sqlite3.lock` в течение `--once` и ручного catch-up. Второй процесс завершается с кодом `75`, не обращаясь к рынку и БД. Lock-файл нельзя удалять во время работы: смена inode разрушит защиту. `--status` только читает состояние.
- `ExecStartPre` проверяет наличие непустой перенесённой БД и запускает локальный `--status --require-baseline`. Без хотя бы одного корректного существующего решения service откажет до сетевого запроса; новый пустой baseline на VPS не создаётся.
- `NoNewPrivileges=true`, `PrivateTmp=true`, `ProtectSystem=strict`, `ProtectHome=true`, `UMask=0077`: процесс не получает новых привилегий, видит отдельный `/tmp`, читает код и `.env`, а запись разрешена только в `/opt/ai-crypto-trader/data` через `ReadWritePaths`. Сеть для публичных свечей доступна. `Restart=no` не создаёт restart storm.

Коды завершения `--once` и сообщения `Outcome:` в journal:

| Код | Категория | Действие оператора |
| --- | --- | --- |
| `0` | `SUCCESS_NEW_DECISION` | Новый час записан. |
| `10` | `SUCCESS_ALREADY_PROCESSED` | Нормальный повтор того же часа. |
| `20` | `NETWORK_TEMPORARY_FAILURE` | Проверить Bybit/сеть; следующий тик timer повторит попытку. |
| `21` | `GAP_REQUIRES_CATCHUP` | Остановить timer, проверить историю и выполнить ручной catch-up. |
| `22` | `DATA_VALIDATION_FAILURE` | Проверить источник свечей и данные; пропуск не маскируется. |
| `23` | `LOCAL_STORAGE_FAILURE` | Проверить доступность SQLite, место и права на `data/`. |
| `24` | `CONFIGURATION_INVALID` | Проверить локальный Demo `.env` и пути. |
| `75` | `SKIPPED_LOCK_BUSY` | Другой shadow-run ещё активен; проверить его журнал и длительность. |

Systemd считает `10` и `75` штатным окончанием (`SuccessExitStatus=10 75`). Остальные ошибки видны в статусе service и journal; timer продолжит свои обычные десятиминутные вызовы. При `21` он не делает catch-up сам. Оператор останавливает timer и разбирает gap вручную. Скрытых retry в market-data клиент не добавлено.

## FIRST INSTALL

1. На Mac прекратите все shadow-команды (`--once`, `--catch-up`, `--status`) и убедитесь, что shadow-процессов больше нет. **После переноса и включения VPS не запускайте live `--once` или `--catch-up` на Mac:** VPS становится единственным активным источником shadow-решений, Mac остаётся средой разработки. Не удаляйте локальную БД.

2. На Mac из каталога проекта подготовьте commit-бандл кода и вычислите SHA-256 **исходной** БД. Bundle переносит зафиксированный код без `.env` и SQLite; это альтернатива загрузке кода через Git remote:

   ```bash
   cd /path/to/ai-crypto-trader
   git status --short
   git bundle create /tmp/ai-crypto-trader.bundle main
   shasum -a 256 data/shadow_trading.sqlite3
   ls -l data/shadow_trading.sqlite3*
   ```

   `git status --short` должен быть пустым. Если рядом остался `-wal` или незавершённый `-journal`, не копируйте один файл БД: сначала разберите активные процессы и сделайте консистентный SQLite backup. Для копирования ниже должны отсутствовать активные записи и незавершённый журнал. Зафиксируйте ожидаемый хеш перед копированием.

3. На Mac загрузите bundle и **этот же файл** БД в отдельные временные имена на VPS:

   ```bash
   scp /tmp/ai-crypto-trader.bundle operator@VPS:/tmp/ai-crypto-trader.bundle
   scp data/shadow_trading.sqlite3 operator@VPS:/tmp/ai-crypto-shadow-transfer.sqlite3
   ```

4. На VPS установите зависимости, создайте отдельного service user, каталог и virtualenv. Каталог `/opt/ai-crypto-trader/` должен быть новым и не принадлежать другому приложению:

   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-venv git sqlite3
   sudo useradd --system --home /nonexistent --shell /usr/sbin/nologin ai-crypto-shadow
   sudo git clone /tmp/ai-crypto-trader.bundle /opt/ai-crypto-trader
   sudo python3 -m venv /opt/ai-crypto-trader/.venv
   sudo /opt/ai-crypto-trader/.venv/bin/python -m pip install -e /opt/ai-crypto-trader
   sudo install -d -m 0700 -o ai-crypto-shadow -g ai-crypto-shadow /opt/ai-crypto-trader/data
   sudo install -m 0600 -o ai-crypto-shadow -g ai-crypto-shadow /opt/ai-crypto-trader/deploy/shadow.env.example /opt/ai-crypto-trader/.env
   ```

   `deploy/shadow.env.example` содержит только Demo URL, пустые поля ключей и отдельные пути; настоящие credentials для shadow не нужны. Не переносите Mac `.env`. Права и владельца `.env`, БД и `data/` держите отдельно от других сервисов.

5. На VPS проверьте SHA-256 временной БД, **сравните его с хешем на Mac**, затем установите файл без создания пустой БД и ещё раз проверьте хеш установленного файла:

   ```bash
   chmod 600 /tmp/ai-crypto-shadow-transfer.sqlite3
   sha256sum /tmp/ai-crypto-shadow-transfer.sqlite3
   sudo install -m 0600 -o ai-crypto-shadow -g ai-crypto-shadow /tmp/ai-crypto-shadow-transfer.sqlite3 /opt/ai-crypto-trader/data/shadow_trading.sqlite3
   sudo sha256sum /opt/ai-crypto-trader/data/shadow_trading.sqlite3
   ```

   **Оба VPS-хеша должны совпасть с записанным Mac-хешем. При расхождении остановитесь; не запускайте service/timer.**

6. На VPS проверьте восстановленное состояние *до* установки timer:

   ```bash
   cd /opt/ai-crypto-trader
   sudo -u ai-crypto-shadow .venv/bin/python -m ai_crypto_trader.diagnostics.shadow_runner --status
   ```

   До переноса отдельно зафиксируйте ожидаемые версию стратегии, состояние, число обработанных свечей, BUY/SELL counts и последнюю UTC-свечу. Сравните с восстановленным снимком; никаких production-значений этот шаблон не содержит. При любом несоответствии timer не включайте. `Health` может уже показывать `GAP` из-за прошедшего времени; это не повод создавать новую пустую БД.

7. Установите units, проверьте их, запустите **один** service run и посмотрите результат. `--once` запишет только одно решение, если новый закрытый час идёт сразу после последнего. При `GAP` он завершится кодом `21` без записи:

   ```bash
   sudo install -m 0644 deploy/systemd/ai-crypto-shadow.service /etc/systemd/system/ai-crypto-shadow.service
   sudo install -m 0644 deploy/systemd/ai-crypto-shadow.timer /etc/systemd/system/ai-crypto-shadow.timer
   sudo systemctl daemon-reload
   sudo systemd-analyze verify /etc/systemd/system/ai-crypto-shadow.service /etc/systemd/system/ai-crypto-shadow.timer
   sudo systemctl start ai-crypto-shadow.service
   systemctl status ai-crypto-shadow.service
   journalctl -u ai-crypto-shadow.service -n 50 --no-pager
   sudo -u ai-crypto-shadow .venv/bin/python -m ai_crypto_trader.diagnostics.shadow_runner --status
   ```

   Если service показал `21`, выполните процедуру MANUAL CATCH-UP ниже только при gap не более `72` часов, затем снова проверьте статус и сделайте один service run. При gap больше `72` часов остановитесь и решите восстановление отдельно. Не включайте timer при `21`, `22`, `23` или `24`.

8. Только после проверки хешей, перенесённого состояния и успешного service run включите timer:

   ```bash
   sudo systemctl enable --now ai-crypto-shadow.timer
   systemctl status ai-crypto-shadow.timer
   ```

## Резервная копия shadow SQLite

Для backup используйте штатную SQLite `.backup`, которая создаёт согласованный снимок даже при открытой БД. Перед update или rollback сначала остановите timer и дождитесь завершения service. Не делайте простой `cp` работающей SQLite. Команда на VPS:

```bash
sudo install -d -m 0700 -o ai-crypto-shadow -g ai-crypto-shadow /opt/ai-crypto-trader/data/backups
sudo -u ai-crypto-shadow sqlite3 /opt/ai-crypto-trader/data/shadow_trading.sqlite3 ".backup '/opt/ai-crypto-trader/data/backups/shadow-$(date -u +%Y%m%dT%H%M%SZ).sqlite3'"
```

Сохраните имя созданного backup и проверьте его отдельно командой `sqlite3 <путь-к-backup> 'PRAGMA integrity_check;'` — ожидается `ok`. Не заменяйте рабочую БД файлом backup при работающем service.

## UPDATE CODE

### Обновление действующего Shadow VPS для Telegram

Это подготовленная процедура для отдельного ручного развёртывания. Текущий
VPS этой задачей не обновляется. Перед работой на Mac создайте и проверьте
Git commit и bundle нового `main`, не включая `.env` или SQLite.

1. На VPS остановите `ai-crypto-shadow.timer` и дождитесь, пока
   `ai-crypto-shadow.service` станет `inactive`; другие сервисы не трогайте.
2. Сделайте SQLite `.backup` действующей `shadow_trading.sqlite3` по разделу
   выше и проверьте `PRAGMA integrity_check`. Запишите исходный Git HEAD.
3. Передайте только bundle кода. Выполните `git fetch` и `merge --ff-only`
   по процедуре UPDATE CODE ниже; **не заменяйте** существующую shadow DB.
4. Обновите `.venv` (`python -m pip install -e /opt/ai-crypto-trader`).
   Добавьте в существующий `/opt/ai-crypto-trader/.env` путь
   `SHADOW_NOTIFICATIONS_PATH=/opt/ai-crypto-trader/data/shadow_notifications.sqlite3`
   и вручную, с правами `0600`, значения `AI_CRYPTO_TELEGRAM_BOT_TOKEN` и
   `AI_CRYPTO_TELEGRAM_CHAT_ID`. Не передавайте секреты через Git или командную
   строку. Новая notification DB создастся отдельно; старую shadow DB не трогайте.
5. Из `/tmp` запустите offline test suite. В каталоге приложения выполните
   `--status`, затем **вручную** `python -m ai_crypto_trader.diagnostics.telegram_notifications --test`
   от пользователя `ai-crypto-shadow`. Проверьте pending/status и один
   systemd service run по процедуре ниже.
6. После успешных проверок запустите timer, убедитесь в штатном journal и
   состояние остальных сервисов сервера не изменилось. При ошибке остановите AI timer и
   разберите причину; не восстанавливайте shadow DB автоматически.

Telegram используется только для `sendMessage`: ни входящих команд, ни
торговых API в service нет. Уникальный event key подавляет обычные дубли;
редкий дубль возможен, если Telegram принял сообщение непосредственно перед
сбоем записи `sent_utc`. Отключение: очистите две Telegram-переменные в `.env`
при остановленном timer, затем перезапустите timer. Shadow-решения продолжаются
без Telegram.

На Mac после проверенного commit создайте новый bundle и передайте его на VPS (как в FIRST INSTALL). На VPS:

1. `sudo systemctl stop ai-crypto-shadow.timer`; дождитесь, пока `systemctl is-active ai-crypto-shadow.service` покажет `inactive` или `failed`. Не обновляйте файлы, пока service работает.
2. Сделайте SQLite backup указанной выше командой; запишите текущий commit: `git -C /opt/ai-crypto-trader rev-parse HEAD`.
3. Выполните `sudo git -C /opt/ai-crypto-trader fetch /tmp/ai-crypto-trader.bundle main` и `sudo git -C /opt/ai-crypto-trader merge --ff-only FETCH_HEAD`. При ошибке fast-forward остановитесь и разберите Git-историю вручную.
4. Обновите пакет: `sudo /opt/ai-crypto-trader/.venv/bin/python -m pip install -e /opt/ai-crypto-trader`. Если менялись шаблоны units, установите их повторно и выполните `sudo systemctl daemon-reload`.
5. Из каталога `/tmp` выполните тесты из `/opt/ai-crypto-trader/tests` установленным virtualenv: `cd /tmp` и `sudo /opt/ai-crypto-trader/.venv/bin/python -m unittest discover -s /opt/ai-crypto-trader/tests -q`. Так рабочий `.env` и БД не подхватываются тестами через текущий каталог.
6. Вернитесь в `cd /opt/ai-crypto-trader` и выполните `sudo -u ai-crypto-shadow .venv/bin/python -m ai_crypto_trader.diagnostics.shadow_runner --status`. Затем один `sudo systemctl start ai-crypto-shadow.service`, проверьте journal/status; только после успеха `sudo systemctl start ai-crypto-shadow.timer`.

## ROLLBACK

Остановите timer, дождитесь завершения service и проверьте доступность backup. Переключите **только код** на записанный до обновления commit: `sudo git -C /opt/ai-crypto-trader switch --detach <KNOWN_GOOD_COMMIT>`, затем переустановите его в том же `.venv`, восстановите соответствующие unit-файлы из каталога `deploy/systemd`, сделайте `sudo systemctl daemon-reload`, повторите тесты и `--status`. БД не откатывайте автоматически: она могла получить новые append-only решения после обновления. Если новая версия изменила схему, восстановление БД требует отдельной проверки совместимости и явного решения оператора. После успешной проверки сделайте один service run и только затем запустите timer.

## STOP SHADOW / START SHADOW

Остановка: `sudo systemctl stop ai-crypto-shadow.timer`, затем дождитесь завершения `ai-crypto-shadow.service` и проверьте `systemctl status ai-crypto-shadow.service`. Для немедленного прерывания зависшего service можно вызвать `sudo systemctl stop ai-crypto-shadow.service`, понимая, что текущая попытка не запишет решение или откатится транзакцией.

Запуск: сначала `--status`, затем один `sudo systemctl start ai-crypto-shadow.service` и проверка journal. Только после успешного one-shot запуска: `sudo systemctl start ai-crypto-shadow.timer`. После reboot enabled timer запустится сам; отдельный service не надо `enable`.

## MANUAL CATCH-UP

Остановите timer и дождитесь завершения service. В `/opt/ai-crypto-trader` выполните от пользователя приложения:

```bash
sudo -u ai-crypto-shadow .venv/bin/python -m ai_crypto_trader.diagnostics.shadow_runner --catch-up --preview
sudo -u ai-crypto-shadow .venv/bin/python -m ai_crypto_trader.diagnostics.shadow_runner --catch-up --execute
sudo -u ai-crypto-shadow .venv/bin/python -m ai_crypto_trader.diagnostics.shadow_runner --status
```

Сверьте границы и количество часов до точного ввода `YES`. Catch-up ограничен `72` часами и записывает весь batch атомарно. Затем запустите один service run, проверьте journal и только после этого снова запустите timer. На Mac после активации VPS эти live-команды выполнять нельзя.

## Health checks и логи

```bash
systemctl status ai-crypto-shadow.timer
systemctl status ai-crypto-shadow.service
journalctl -u ai-crypto-shadow.service -n 50 --no-pager
cd /opt/ai-crypto-trader
sudo -u ai-crypto-shadow .venv/bin/python -m ai_crypto_trader.diagnostics.shadow_runner --status
```

Если кратковременный timeout совпал с тиком timer, новая запись не появляется, а следующий тик повторяет `--once`. После смены закрытого часа пропуск обнаруживается как `GAP` и не исправляется автоматически. Логи показывают категорию отказа без API Key/Secret, подписи и ответа аккаунта.
