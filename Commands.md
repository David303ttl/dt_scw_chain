
-----

````markdown
### Commands

**1. Clone the repository:**

```bash
git clone [https://github.com/David303ttl/dt_scw_chain.git](https://github.com/David303ttl/dt_scw_chain.git) 
cd DT_SCW_CHAIN
````

**2. Install dependencies (once after cloning):**

```bash
poetry install
```

**3. Run the main script for SCW conversion and chain creation:**

```bash
poetry run python batch_convert_scw.py
```

**4. Run the script for WaveEdit banks conversion:**

```bash
poetry run python batch_waveedit_banks.py
```

**5. Run the script for image to wavetable conversion:**

```bash
poetry run python batch_image_wavetables.py
```

**Additional, but useful Poetry commands:**

  * **Check environment status:**
    ```bash
    poetry env list
    poetry env info
    ```
  * **Add a new dependency:**
    ```bash
    poetry add library_name
    ```
  * **Update dependencies:**
    ```bash
    poetry update
    ```
  * **Remove the virtual environment (if you want to start fresh):**
    ```bash
    poetry env remove --all
    ```

```
```
