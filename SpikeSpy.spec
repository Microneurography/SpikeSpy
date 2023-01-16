# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

binaries = [] # collect_dynamic_libs('numpy')
datas = collect_data_files('nixio')
datas += [('spikespy/ui/icon.*','.')]
a = Analysis(['run.py'],
             pathex=[],
             binaries=binaries,
             datas=datas,
             hiddenimports=['nixio', 'scipy','scipy.signal'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=['pandas'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='SpikeSpy',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
