How to make a new release?
--------------------------

1. Update the ``mitsuba`` dependency version in ``pyproject.toml`` if required.

2. Ensure that the changelog is up to date in ``docs/src/release_notes.rst``.

3. Verify that the CI is currently green.

4. Run the Github Action "Build Python Wheels" with option "0". This effectively is a dry
   run of the wheel creation process.

5. If the action failed, fix whatever broke in the build process. If it succeded
   continue.

6. Add release number and date to ``docs/src/release_notes.rst``.

7. Commit: ``git commit -am "vX.Y.Z release"``

8. Tag: ``git tag -a vX.Y.Z -m "vX.Y.Z release"``

9. Push: ``git push`` and ``git push --tags``

10. Run the GHA "Build Python Wheels" with option "1".

11. Check that the new version is available on
    `readthedocs <https://drtvam.readthedocs.io/en/latest/>`__.

12. Create a `release on GitHub <https://github.com/rgl-epfl/drtvam/releases/new>`__
    from the tag created at step 10. The changelog can be copied from step 2.

13. Checkout the ``stable`` branch and run ``git pull --ff-only origin vX.Y.Z``
    and ``git push``
