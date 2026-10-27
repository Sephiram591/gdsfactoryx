"""Delete a cell from the layout."""

if __name__ == "__main__":
    import gdsfactory as gf

    gf.gpdk.PDK.activate()

    c = gf.Component()
    ref1 = c << gf.components.text("1")
    ref2 = c << gf.components.text("2")

    print("Cells before:", len(c.called_cells()))

    # Delete a specific cell: it is dropped from the cache and from its parents
    ref2.cell.delete()

    print("Cells after:", len(c.called_cells()))
    c.show()

    # Or clear the entire cache (all cells)
    # gf.clear_cache()
