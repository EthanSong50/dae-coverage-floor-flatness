# surface_profiling/utils/heatmap_generator.py

import os
import open3d as o3d
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 비활성화 (헤드리스 환경 안전성 확보)
import matplotlib.pyplot as plt
import yaml
import matplotlib.image as mpimg


def _load_occupancy_map(map_yaml_dir):
    """map_yaml_dir(.yaml파일이 있는 위치)와 같은 폴더의 이미지(.pgm/.png)를 읽어
    (image_array, resolution, origin_x, origin_y)를 반환함.
    origin은 ROS map_server 규격 그대로: 이미지 '왼쪽 아래' 픽셀이 world 좌표
    (origin_x, origin_y)에 대응함.

    주의: origin의 세 번째 값(yaw)은 현재 반영하지 않음(0으로 가정).
    yaw != 0인 맵을 쓰게 되면 히트맵과 배경 맵이 어긋나 보이므로,
    그 경우 이 함수에 회전 변환을 추가해야 함."""
    with open(map_yaml_dir, 'r') as f:
        meta = yaml.safe_load(f)

    image_path = meta['image']
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(map_yaml_dir), image_path)

    # 1. 원본 이미지 로드
    img = mpimg.imread(image_path)
    
    # 2. [수정] 행렬의 상하를 반전(Flip Up-Down)시켜 origin='lower' 좌표계와 동기화
    img = np.flipud(img)

    resolution = meta['resolution']
    origin = meta.get('origin', [0.0, 0.0, 0.0])
    yaw = origin[2] if len(origin) > 2 else 0.0
    if abs(yaw) > 1e-6:
        import warnings
        warnings.warn(
            f"[heatmap_generator] map origin의 yaw={yaw}(rad)가 0이 아닙니다. "
            "현재 구현은 yaw=0을 가정하고 있어 히트맵과 배경 맵이 어긋날 수 있습니다."
        )
    return img, resolution, origin[0], origin[1]


def generate_floor_heatmap(
    pcd_path,
    output_img_path,
    grid_size=0.02,
    z_min=-0.005,
    z_max=0.035,
    map_yaml_dir=None,
):
    """
    바닥면 필터링이 끝난 PCD 데이터를 X-Y 평면 격자로 나누고, 각 격자의 평균 z값을
    바닥 요철 실측 범위([z_min, z_max])를 기준으로 정규화해 평탄도 히트맵
    이미지를 생성, PNG로 저장함.

    z_min, z_max: 색상 스케일의 절대 하한/상한(m). 데이터셋 자체의 min/max가
        아니라 "바닥 평탄도로서 의미 있는 범위"를 고정값으로 지정함 —
        벽처럼 이 범위를 벗어나는 포인트가 섞여 들어와도 색 스케일이
        영향받지 않도록 하기 위함. floor_extractor의 z_min/z_max와
        동일한 값을 쓰는 것을 권장.
    map_yaml_dir: 지정하면(예: map_from_dae.yaml) 해당 2D OGM 이미지를
        배경으로 깔고 그 위에 히트맵을 반투명 오버레이함. None이면
        히트맵만 단독으로 그림(하위 호환).

    참고: 정규화는 이제 데이터셋 자체의 min/max가 아니라 z_min/z_max로 고정된
    절대 기준이며, 여전히 건축 표준 규격에 따른 평탄도 허용 오차 기준과는
    별개의 표현임에 유의.
    """
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"[-] PCD file not found: {pcd_path}")

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    if points.shape[0] == 0:
        raise ValueError("[-] No points found in PCD. Cannot generate heatmap.")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # 격자 생성
    x_bins = np.arange(x.min(), x.max() + grid_size, grid_size)
    y_bins = np.arange(y.min(), y.max() + grid_size, grid_size)

    # 2D 히스토그램으로 z값 평균 계산
    heatmap_sum, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=z)
    heatmap_count, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    heatmap = np.divide(heatmap_sum, heatmap_count, out=np.zeros_like(heatmap_sum), where=heatmap_count != 0)
    heatmap[heatmap_count == 0] = np.nan  # 빈 공간은 NaN 처리

    # 데이터셋 자체 min/max가 아니라, 바닥 평탄도로서 의미 있는
    # 절대 범위(z_min~z_max)로 정규화 + clip. 이렇게 해야 필터링을 뚫고
    # 섞여 들어온 극단값(예: 벽 하단 일부)이 있어도 색 스케일이 왜곡되지 않음.
    heatmap_clipped = np.clip(heatmap, z_min, z_max)
    heatmap_norm = (heatmap_clipped - z_min) / (z_max - z_min)

    # 히트맵 시각화
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap('jet')
    cmap.set_bad(color=(1, 1, 1, 0) if map_yaml_dir else 'white')  # 맵 오버레이 시 빈 공간은 완전 투명

    extent = [x.min(), x.max(), y.min(), y.max()]

    if map_yaml_dir is not None:
        if os.path.exists(map_yaml_dir):
            # 2D OGM을 배경으로 깔고, 그 위에 히트맵을 반투명 오버레이
            map_img, resolution, origin_x, origin_y = _load_occupancy_map(map_yaml_dir)
            map_h, map_w = map_img.shape[:2]
            map_extent = [
                origin_x,
                origin_x + map_w * resolution,
                origin_y,
                origin_y + map_h * resolution,
            ]
            ax.imshow(map_img, cmap='gray', origin='lower', extent=map_extent, zorder=0)
            im = ax.imshow(
                heatmap_norm.T,
                origin='lower',
                extent=extent,
                cmap=cmap,
                interpolation='nearest',
                alpha=0.75,
                zorder=1,
            )
            # 맵 좌표계를 기준으로 뷰를 맞춘다(히트맵이 맵보다 작은 경우가 일반적).
            ax.set_xlim(map_extent[0], map_extent[1])
            ax.set_ylim(map_extent[2], map_extent[3])
        else:
            print(f"[!] Warning: map_yaml_dir가 제공되었으나 파일을 찾을 수 없습니다: {map_yaml_dir}")
            im = ax.imshow(
                heatmap_norm.T,
                origin='lower',
                extent=extent,
                cmap=cmap,
                interpolation='nearest',
            )
    else:
        print(f"[!] Warning: map_yaml_dir가 제공되지 않았습니다: {map_yaml_dir}")
        im = ax.imshow(
            heatmap_norm.T,
            origin='lower',
            extent=extent,
            cmap=cmap,
            interpolation='nearest',
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'Height (Z) [cm], scale fixed to [{z_min*100:.1f}, {z_max*100:.1f}]')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Floor Flatness Heatmap (Top-down View)')
    ax.set_aspect('equal')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[+] Heatmap saved to: {output_img_path}")
    return output_img_path